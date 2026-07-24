## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 43.3827531155


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.6606002, 28.3044319, -7.6606002, 28.3044319, -35.9650307, 35.9650307)
1: (-8.9601765, 31.9744511, -8.9601765, 31.9744511, -40.9346275, 40.9346275)
2: (-9.4893942, 32.1866379, -9.4893942, 32.1866379, -41.6760216, 41.6760216)
3: (-14.0368652, 32.8534927, -14.0368652, 32.8534927, -46.8903580, 46.8903580)
4: (-14.5853920, 32.1437492, -14.5853920, 32.1437492, -46.7291412, 46.7291412)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.62 + 1.43 = 3.05 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -43.6007569, upper bound: 43.6007569

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5760261, upper bound: 43.5949019
time: 0.51 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.43 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.08 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 1.08
Output dim: 3, lower bound: -43.5760261, upper bound: 43.5949019
NS_B2, status: Status.UNKNOWN, split count: 1, time: 1.08
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -7.6606002, 28.3044319, -6.4284191, 24.1388493, -31.7994499, 34.7328491
1: -8.9601765, 31.9744511, -7.4645872, 27.3031864, -36.2633629, 39.4390373
2: -9.4893942, 32.1866379, -7.9618182, 27.3991623, -36.8885536, 40.1484489
3: -14.0368652, 32.8534927, -11.7890568, 27.9809723, -42.0178375, 44.6425476
4: -14.5853920, 32.1437492, -12.4559155, 27.1424770, -41.7278671, 44.5996590

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.81 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.48 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -7.6606002, 28.3044319, -8.8551903, 33.0941162, -40.7547150, 37.1596222
1: -8.9601765, 31.9744511, -10.4291992, 37.5969582, -46.5571365, 42.4036484
2: -9.4893942, 32.1866379, -10.9747181, 37.6314697, -47.1208649, 43.1613541
3: -14.0368652, 32.8534927, -16.3464375, 38.8504906, -52.8873520, 49.1999283
4: -14.5853920, 32.1437492, -17.2999840, 37.0951767, -51.6805687, 49.4437332

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.39 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
time: 0.39 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.42 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 2.42
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 2.42
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 2.42
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 2.42
Output dim: 3, lower bound: -43.5701711, upper bound: 43.5701711

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -6.4284191, 24.1388493, -6.4284191, 24.1388493, -30.5672684, 30.5672684
1: -7.4645872, 27.3031864, -7.4645872, 27.3031864, -34.7677727, 34.7677727
2: -7.9618182, 27.3991623, -7.9618182, 27.3991623, -35.3609810, 35.3609810
3: -11.7890568, 27.9809723, -11.7890568, 27.9809723, -39.7700233, 39.7700272
4: -12.4559155, 27.1424770, -12.4559155, 27.1424770, -39.5983925, 39.5983925

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B1_A1_B1

### Relational analysis result of NS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5760261, upper bound: 43.5948731
time: 0.42 seconds

## Relational analysis of NS_B1_A1_B2

### Relational analysis result of NS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5723781, upper bound: 43.5787431
time: 0.51 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -8.8551903, 33.0941162, -6.4284191, 24.1388493, -32.9940414, 39.5225296
1: -10.4291992, 37.5969582, -7.4645872, 27.3031864, -37.7323837, 45.0615425
2: -10.9747181, 37.6314697, -7.9618182, 27.3991623, -38.3738785, 45.5932884
3: -16.3464375, 38.8504906, -11.7890568, 27.9809723, -44.3274078, 50.6395416
4: -17.2999840, 37.0951767, -12.4559155, 27.1424770, -44.4424591, 49.5510902

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5760261, upper bound: 43.5949019
time: 0.43 seconds

## Relational analysis of NS_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5726843, upper bound: 43.5814188
time: 0.45 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -6.4284191, 24.1388493, -8.8551903, 33.0941162, -39.5225296, 32.9940414
1: -7.4645872, 27.3031864, -10.4291992, 37.5969582, -45.0615425, 37.7323837
2: -7.9618182, 27.3991623, -10.9747181, 37.6314697, -45.5932884, 38.3738785
3: -11.7890568, 27.9809723, -16.3464375, 38.8504906, -50.6395416, 44.3274078
4: -12.4559155, 27.1424770, -17.2999840, 37.0951767, -49.5510902, 44.4424591

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A1_B1

### Relational analysis result of NS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5442300, upper bound: 43.5658637
time: 0.43 seconds

## Relational analysis of NS_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5409015, upper bound: 43.5409015
time: 0.43 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -8.8551903, 33.0941162, -8.8551903, 33.0941162, -41.9493065, 41.9493065
1: -10.4291992, 37.5969582, -10.4291992, 37.5969582, -48.0261536, 48.0261536
2: -10.9747181, 37.6314697, -10.9747181, 37.6314697, -48.6061859, 48.6061859
3: -16.3464375, 38.8504906, -16.3464375, 38.8504906, -55.1969299, 55.1969299
4: -17.2999840, 37.0951767, -17.2999840, 37.0951767, -54.3951607, 54.3951607

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_A1

### Relational analysis result of NS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5658637, upper bound: 43.5442300
time: 0.46 seconds

## Relational analysis of NS_B2_A2_A2

### Relational analysis result of NS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5409015, upper bound: 43.5409015
time: 0.46 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.55 seconds
NS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 3, lower bound: -43.5760261, upper bound: 43.5948731
NS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 3, lower bound: -43.5723781, upper bound: 43.5787431
NS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 3, lower bound: -43.5760261, upper bound: 43.5949019
NS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 3, lower bound: -43.5726843, upper bound: 43.5814188
NS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 3, lower bound: -43.5442300, upper bound: 43.5658637
NS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 3, lower bound: -43.5409015, upper bound: 43.5409015
NS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 3, lower bound: -43.5658637, upper bound: 43.5442300
NS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 3, lower bound: -43.5409015, upper bound: 43.5409015

## BFS NS instance: NS_B1_A1_B1

### Backsubstitution after applying NS history:
0: -6.4284191, 24.1388493, -6.0080776, 22.7014885, -29.1299057, 30.1469250
1: -7.4645872, 27.3031864, -6.9569273, 25.7120094, -33.1765900, 34.2601128
2: -7.9618182, 27.3991623, -7.4386706, 25.7422943, -33.7041130, 34.8378334
3: -11.7890568, 27.9809723, -11.0225744, 26.3203468, -38.1093941, 39.0035477
4: -12.4559155, 27.1424770, -11.7293959, 25.3994999, -37.8554077, 38.8718719

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B1_A1_B1_A1

### Relational analysis result of NS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5809501, upper bound: 43.5809501
time: 0.46 seconds

## Relational analysis of NS_B1_A1_B1_A2

### Relational analysis result of NS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5809501, upper bound: 43.5809501
time: 0.49 seconds

## BFS NS instance: NS_B1_A1_B2

### Backsubstitution after applying NS history:
0: -6.4284191, 24.1388493, -6.5748506, 24.7860813, -31.2145004, 30.7136993
1: -7.4645872, 27.3031864, -7.6228886, 28.1002808, -35.5648689, 34.9260750
2: -7.9618182, 27.3991623, -8.1419983, 28.1216526, -36.0834694, 35.5411606
3: -11.7890568, 27.9809723, -12.0498886, 28.8040485, -40.5931015, 40.0308609
4: -12.4559155, 27.1424770, -12.8321266, 27.7175007, -40.1734085, 39.9746017

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B1_A1_B2_A1

### Relational analysis result of NS_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5809501, upper bound: 43.5809501
time: 0.47 seconds

## Relational analysis of NS_B1_A1_B2_A2

### Relational analysis result of NS_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5809501, upper bound: 43.5809501
time: 0.49 seconds

## BFS NS instance: NS_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.8551903, 33.0941162, -5.2551436, 20.1315994, -28.9867897, 38.3492584
1: -10.4291992, 37.5969582, -6.0467672, 22.8425674, -33.2717590, 43.6437263
2: -10.9747181, 37.6314697, -6.4991312, 22.7556610, -33.7303772, 44.1306000
3: -16.3464375, 38.8504906, -9.6570997, 23.3396664, -39.6861038, 48.5075874
4: -17.2999840, 37.0951767, -10.4297295, 22.2815781, -39.5815620, 47.5249062

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A2_B1_A1

### Relational analysis result of NS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5726437, upper bound: 43.5725862
time: 0.42 seconds

## Relational analysis of NS_B1_A2_B1_A2

### Relational analysis result of NS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5476815, upper bound: 43.5692578
time: 0.43 seconds

## BFS NS instance: NS_B1_A2_B2

### Backsubstitution after applying NS history:
0: -8.8551903, 33.0941162, -6.7437096, 25.4687443, -34.3239365, 39.8378220
1: -10.4291992, 37.5969582, -7.8409715, 28.8711681, -39.3003616, 45.4379311
2: -10.9747181, 37.6314697, -8.3597012, 28.9183235, -39.8930435, 45.9911652
3: -16.3464375, 38.8504906, -12.3940144, 29.6448441, -45.9912758, 51.2445068
4: -17.2999840, 37.0951767, -13.1956530, 28.4903889, -45.7903748, 50.2908287

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A2_B2_A1

### Relational analysis result of NS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5692031, upper bound: 43.5586702
time: 0.53 seconds

## Relational analysis of NS_B1_A2_B2_A2

### Relational analysis result of NS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5442409, upper bound: 43.5553418
time: 0.41 seconds

## BFS NS instance: NS_B2_A1_B1

### Backsubstitution after applying NS history:
0: -6.4284191, 24.1388493, -8.6322088, 32.3104172, -38.7388306, 32.7710571
1: -7.4645872, 27.3031864, -10.1549702, 36.7054749, -44.1700592, 37.4581566
2: -7.9618182, 27.3991623, -10.6975584, 36.7274017, -44.6892204, 38.0967178
3: -11.7890568, 27.9809723, -15.9301043, 37.9201851, -49.7092400, 43.9110756
4: -12.4559155, 27.1424770, -16.8876266, 36.1860275, -48.6419373, 44.0301056

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_A1_B1_A1

### Relational analysis result of NS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5692578, upper bound: 43.5476815
time: 0.70 seconds

## Relational analysis of NS_B2_A1_B1_A2

### Relational analysis result of NS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5692578, upper bound: 43.5476815
time: 0.66 seconds

## BFS NS instance: NS_B2_A1_B2

### Backsubstitution after applying NS history:
0: -6.4284191, 24.1388493, -8.8174801, 33.0339127, -39.4623260, 32.9563293
1: -7.4645872, 27.3031864, -10.3915501, 37.5448494, -45.0094337, 37.6947365
2: -7.9618182, 27.3991623, -10.9315701, 37.5578690, -45.5196877, 38.3307343
3: -11.7890568, 27.9809723, -16.3067951, 38.8084564, -50.5975113, 44.2877617
4: -12.4559155, 27.1424770, -17.2786331, 36.9893837, -49.4452972, 44.4211121

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_A1_B2_B1

### Relational analysis result of NS_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5218612, upper bound: 43.4160098
time: 0.61 seconds

## Relational analysis of NS_B2_A1_B2_B2

### Relational analysis result of NS_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5691286, upper bound: 43.5475238
time: 0.50 seconds

## BFS NS instance: NS_B2_A2_A1

### Backsubstitution after applying NS history:
0: -8.6322088, 32.3104172, -8.8551903, 33.0941162, -41.7263222, 41.1656075
1: -10.1549702, 36.7054749, -10.4291992, 37.5969582, -47.7519264, 47.1346703
2: -10.6975584, 36.7274017, -10.9747181, 37.6314697, -48.3290291, 47.7021179
3: -15.9301043, 37.9201851, -16.3464375, 38.8504906, -54.7805939, 54.2666245
4: -16.8876266, 36.1860275, -17.2999840, 37.0951767, -53.9828033, 53.4860115

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B2_A2_A1_B1

### Relational analysis result of NS_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4317113, upper bound: 43.4934255
time: 0.48 seconds

## Relational analysis of NS_B2_A2_A1_B2

### Relational analysis result of NS_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3821234, upper bound: 43.3843393
time: 0.47 seconds

## BFS NS instance: NS_B2_A2_A2

### Backsubstitution after applying NS history:
0: -8.8174801, 33.0339127, -8.8551903, 33.0941162, -41.9115982, 41.8891029
1: -10.3915501, 37.5448494, -10.4291992, 37.5969582, -47.9885063, 47.9740448
2: -10.9315701, 37.5578690, -10.9747181, 37.6314697, -48.5630417, 48.5325851
3: -16.3067951, 38.8084564, -16.3464375, 38.8504906, -55.1572800, 55.1548920
4: -17.2786331, 36.9893837, -17.2999840, 37.0951767, -54.3738098, 54.2893677

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_A2_A1

### Relational analysis result of NS_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
time: 0.42 seconds

## Relational analysis of NS_B2_A2_A2_A2

### Relational analysis result of NS_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476
time: 0.68 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.77 seconds
NS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 3, lower bound: -43.5809501, upper bound: 43.5809501
NS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 3, lower bound: -43.5809501, upper bound: 43.5809501
NS_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 3, lower bound: -43.5809501, upper bound: 43.5809501
NS_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 3, lower bound: -43.5809501, upper bound: 43.5809501
NS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 3, lower bound: -43.5726437, upper bound: 43.5725862
NS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 3, lower bound: -43.5476815, upper bound: 43.5692578
NS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 3, lower bound: -43.5692031, upper bound: 43.5586702
NS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 3, lower bound: -43.5442409, upper bound: 43.5553418
NS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 3, lower bound: -43.5692578, upper bound: 43.5476815
NS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 3, lower bound: -43.5692578, upper bound: 43.5476815
NS_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 3, lower bound: -43.5218612, upper bound: 43.4160098
NS_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 3, lower bound: -43.5691286, upper bound: 43.5475238
NS_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 3, lower bound: -43.4317113, upper bound: 43.4934255
NS_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 3, lower bound: -43.3821234, upper bound: 43.3843393
NS_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 3, lower bound: -43.5387238, upper bound: 43.5154913
NS_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 3, lower bound: -43.5134476, upper bound: 43.5134476

## BFS NS instance: NS_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -6.0080776, 22.7014885, -6.0080776, 22.7014885, -28.7095642, 28.7095642
1: -6.9569273, 25.7120094, -6.9569273, 25.7120094, -32.6689377, 32.6689339
2: -7.4386706, 25.7422943, -7.4386706, 25.7422943, -33.1809654, 33.1809654
3: -11.0225744, 26.3203468, -11.0225744, 26.3203468, -37.3429146, 37.3429146
4: -11.7293959, 25.3994999, -11.7293959, 25.3994999, -37.1288948, 37.1288948

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_B1_A1_B1

### Relational analysis result of NS_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5605188, upper bound: 43.5922586
time: 0.47 seconds

## Relational analysis of NS_B1_A1_B1_A1_B2

### Relational analysis result of NS_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5581766, upper bound: 43.5716737
time: 0.43 seconds

## BFS NS instance: NS_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -6.5748506, 24.7860813, -6.0080776, 22.7014885, -29.2763386, 30.7941570
1: -7.6228886, 28.1002808, -6.9569273, 25.7120094, -33.3348923, 35.0572090
2: -8.1419983, 28.1216526, -7.4386706, 25.7422943, -33.8842926, 35.5603218
3: -12.0498886, 28.8040485, -11.0225744, 26.3203468, -38.3702354, 39.8266220
4: -12.8321266, 27.7175007, -11.7293959, 25.3994999, -38.2316208, 39.4468956

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_B1_A2_A1

### Relational analysis result of NS_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5811461, upper bound: 43.5747232
time: 0.47 seconds

## Relational analysis of NS_B1_A1_B1_A2_A2

### Relational analysis result of NS_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5581766, upper bound: 43.5716737
time: 0.45 seconds

## BFS NS instance: NS_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6.0080776, 22.7014885, -6.5748506, 24.7860813, -30.7941570, 29.2763386
1: -6.9569273, 25.7120094, -7.6228886, 28.1002808, -35.0572090, 33.3348999
2: -7.4386706, 25.7422943, -8.1419983, 28.1216526, -35.5603218, 33.8842926
3: -11.0225744, 26.3203468, -12.0498886, 28.8040485, -39.8266220, 38.3702354
4: -11.7293959, 25.3994999, -12.8321266, 27.7175007, -39.4468956, 38.2316208

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_B2_A1_B1

### Relational analysis result of NS_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5569962, upper bound: 43.5769161
time: 0.44 seconds

## Relational analysis of NS_B1_A1_B2_A1_B2

### Relational analysis result of NS_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5539466, upper bound: 43.5539466
time: 0.44 seconds

## BFS NS instance: NS_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -6.5748506, 24.7860813, -6.5748506, 24.7860813, -31.3609314, 31.3609314
1: -7.6228886, 28.1002808, -7.6228886, 28.1002808, -35.7231674, 35.7231674
2: -8.1419983, 28.1216526, -8.1419983, 28.1216526, -36.2636490, 36.2636490
3: -12.0498886, 28.8040485, -12.0498886, 28.8040485, -40.8539352, 40.8539352
4: -12.8321266, 27.7175007, -12.8321266, 27.7175007, -40.5496216, 40.5496216

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_B2_A2_B1

### Relational analysis result of NS_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5737330, upper bound: 43.5782946
time: 0.73 seconds

## Relational analysis of NS_B1_A1_B2_A2_B2

### Relational analysis result of NS_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5722960, upper bound: 43.5722960
time: 0.43 seconds

## BFS NS instance: NS_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -8.6322088, 32.3104172, -5.2551436, 20.1315994, -28.7638092, 37.5655594
1: -10.1549702, 36.7054749, -6.0467672, 22.8425674, -32.9975319, 42.7522430
2: -10.6975584, 36.7274017, -6.4991312, 22.7556610, -33.4532204, 43.2265320
3: -15.9301043, 37.9201851, -9.6570997, 23.3396664, -39.2697716, 47.5772858
4: -16.8876266, 36.1860275, -10.4297295, 22.2815781, -39.1692047, 46.6157532

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B1_A1_B1

### Relational analysis result of NS_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5604773, upper bound: 43.5721456
time: 0.44 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2

### Relational analysis result of NS_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5557536, upper bound: 43.5263719
time: 0.41 seconds

## BFS NS instance: NS_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8.8174801, 33.0339127, -5.2551436, 20.1315994, -28.9490757, 38.2890549
1: -10.3915501, 37.5448494, -6.0467672, 22.8425674, -33.2341118, 43.5916176
2: -10.9315701, 37.5578690, -6.4991312, 22.7556610, -33.6872330, 44.0569992
3: -16.3067951, 38.8084564, -9.6570997, 23.3396664, -39.6464577, 48.4655533
4: -17.2786331, 36.9893837, -10.4297295, 22.2815781, -39.5602112, 47.4191132

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5224054, upper bound: 43.5670800
time: 0.42 seconds

## Relational analysis of NS_B1_A2_B1_A2_B2

### Relational analysis result of NS_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5176817, upper bound: 43.5213063
time: 0.44 seconds

## BFS NS instance: NS_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -8.6322088, 32.3104172, -6.7437096, 25.4687443, -34.1009483, 39.0541191
1: -10.1549702, 36.7054749, -7.8409715, 28.8711681, -39.0261307, 44.5464478
2: -10.6975584, 36.7274017, -8.3597012, 28.9183235, -39.6158791, 45.0870972
3: -15.9301043, 37.9201851, -12.3940144, 29.6448441, -45.5749435, 50.3142014
4: -16.8876266, 36.1860275, -13.1956530, 28.4903889, -45.3780136, 49.3816795

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A2_B2_A1_B1

### Relational analysis result of NS_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5442409, upper bound: 43.5553418
time: 0.40 seconds

## Relational analysis of NS_B1_A2_B2_A1_B2

### Relational analysis result of NS_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5442409, upper bound: 43.5553418
time: 0.45 seconds

## BFS NS instance: NS_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.8174801, 33.0339127, -6.7437096, 25.4687443, -34.2862244, 39.7776146
1: -10.3915501, 37.5448494, -7.8409715, 28.8711681, -39.2627106, 45.3858223
2: -10.9315701, 37.5578690, -8.3597012, 28.9183235, -39.8498917, 45.9175644
3: -16.3067951, 38.8084564, -12.3940144, 29.6448441, -45.9516296, 51.2024689
4: -17.2786331, 36.9893837, -13.1956530, 28.4903889, -45.7690201, 50.1850357

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A2_B2_A2_B1

### Relational analysis result of NS_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5442409, upper bound: 43.5553418
time: 0.47 seconds

## Relational analysis of NS_B1_A2_B2_A2_B2

### Relational analysis result of NS_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5442409, upper bound: 43.5553418
time: 0.48 seconds

## BFS NS instance: NS_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -6.2246099, 23.4275513, -8.6322088, 32.3104172, -38.5350266, 32.0597534
1: -7.2165470, 26.4996204, -10.1549702, 36.7054749, -43.9220200, 36.6545906
2: -7.7077212, 26.5764866, -10.6975584, 36.7274017, -44.4351196, 37.2740364
3: -11.4118147, 27.1407299, -15.9301043, 37.9201851, -49.3319931, 43.0708351
4: -12.0827684, 26.3105183, -16.8876266, 36.1860275, -48.2687912, 43.1981430

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_B1_A1_A1

### Relational analysis result of NS_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5721456, upper bound: 43.5604773
time: 0.48 seconds

## Relational analysis of NS_B2_A1_B1_A1_A2

### Relational analysis result of NS_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5263719, upper bound: 43.5557536
time: 0.49 seconds

## BFS NS instance: NS_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -6.3933344, 24.0808926, -8.6322088, 32.3104172, -38.7037468, 32.7130928
1: -7.4286666, 27.2350445, -10.1549702, 36.7054749, -44.1341400, 37.3900070
2: -7.9199948, 27.3296623, -10.6975584, 36.7274017, -44.6473961, 38.0272141
3: -11.7527027, 27.9425888, -15.9301043, 37.9201851, -49.6728821, 43.8726921
4: -12.4385681, 27.0379066, -16.8876266, 36.1860275, -48.6245956, 43.9255333

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_B1_A2_A1

### Relational analysis result of NS_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5721456, upper bound: 43.5604773
time: 0.45 seconds

## Relational analysis of NS_B2_A1_B1_A2_A2

### Relational analysis result of NS_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5263719, upper bound: 43.5557536
time: 0.79 seconds

## BFS NS instance: NS_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -6.4284191, 24.1388493, -8.7770147, 32.8421860, -39.2706070, 32.9158630
1: -7.4645872, 27.3031864, -10.3439817, 37.2919693, -44.7565536, 37.6471672
2: -7.9618182, 27.3991623, -10.8761139, 37.3595619, -45.3213806, 38.2752762
3: -11.7890568, 27.9809723, -16.2282066, 38.5262871, -50.3153419, 44.2091789
4: -12.4559155, 27.1424770, -17.1341934, 36.8623123, -49.3182220, 44.2766724

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## BFS NS instance: NS_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -6.4284191, 24.1388493, -8.8129749, 33.0192146, -39.4476280, 32.9518242
1: -7.4645872, 27.3031864, -10.3859720, 37.5279694, -44.9925537, 37.6891594
2: -7.9618182, 27.3991623, -10.9260187, 37.5408058, -45.5026207, 38.3251762
3: -11.7890568, 27.9809723, -16.2985821, 38.7907181, -50.5797653, 44.2795563
4: -12.4559155, 27.1424770, -17.2704239, 36.9720306, -49.4279404, 44.4129028

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_A1_B2_B2_A1

### Relational analysis result of NS_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5691286, upper bound: 43.5475238
time: 0.58 seconds

## Relational analysis of NS_B2_A1_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -8.6322088, 32.3104172, -8.8156204, 32.9617119, -41.5939140, 41.1260338
1: -10.1549702, 36.7054749, -10.3809900, 37.4477654, -47.6027336, 47.0864639
2: -10.6975584, 36.7274017, -10.9258604, 37.4781151, -48.1756706, 47.6532516
3: -15.9301043, 37.9201851, -16.2741280, 38.6946754, -54.6247787, 54.1943130
4: -16.8876266, 36.1860275, -17.2310925, 36.9354782, -53.8231049, 53.4171143

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_A1_B1_B1

### Relational analysis result of NS_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4317113, upper bound: 43.4934255
time: 0.51 seconds

## Relational analysis of NS_B2_A2_A1_B1_B2

### Relational analysis result of NS_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4317113, upper bound: 43.4934255
time: 1.48 seconds

## BFS NS instance: NS_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -8.6322088, 32.3104172, -9.3646259, 35.1735840, -43.8057899, 41.6750298
1: -10.1549702, 36.7054749, -11.0751286, 40.0641594, -50.1880035, 47.7806015
2: -10.6975584, 36.7274017, -11.6250591, 39.9605408, -50.6580963, 48.3524628
3: -15.9301043, 37.9201851, -17.3567867, 41.4715424, -57.3551712, 55.2769699
4: -16.8876266, 36.1860275, -18.4965115, 39.2655792, -56.1532059, 54.6825409

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_A1_B2_B1

### Relational analysis result of NS_B2_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3538075, upper bound: 43.3432026
time: 0.64 seconds

## Relational analysis of NS_B2_A2_A1_B2_B2

### Relational analysis result of NS_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3818490, upper bound: 43.3840848
time: 0.48 seconds

## BFS NS instance: NS_B2_A2_A2_A1

### Backsubstitution after applying NS history:
0: -8.5439262, 32.0631142, -8.8551903, 33.0941162, -41.6380424, 40.9183044
1: -10.0602074, 36.4324837, -10.4291992, 37.5969582, -47.6571655, 46.8616791
2: -10.5933409, 36.4442482, -10.9747181, 37.6314697, -48.2248116, 47.4189682
3: -15.7987118, 37.6562881, -16.3464375, 38.8504906, -54.6492004, 54.0027237
4: -16.7670708, 35.8889656, -17.2999840, 37.0951767, -53.8622475, 53.1889496

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B2_A2_A2_A1_B1

### Relational analysis result of NS_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4275308, upper bound: 43.4514684
time: 0.70 seconds

## Relational analysis of NS_B2_A2_A2_A1_B2

### Relational analysis result of NS_B2_A2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3798398, upper bound: 43.3575275
time: 0.51 seconds

## BFS NS instance: NS_B2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -9.1115875, 33.9125404, -8.8551903, 33.0941162, -42.2057037, 42.7677307
1: -10.7727814, 38.5158424, -10.4291992, 37.5969582, -48.3697395, 48.9450378
2: -11.2928276, 38.6079216, -10.9747181, 37.6314697, -48.9242973, 49.5826416
3: -16.8676147, 39.8401985, -16.3464375, 38.8504906, -55.7181015, 56.1866341
4: -17.7444897, 38.1677361, -17.2999840, 37.0951767, -54.8396683, 55.4677200

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B2_A2_A2_A2_B1

### Relational analysis result of NS_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4195067, upper bound: 43.4657565
time: 0.49 seconds

## Relational analysis of NS_B2_A2_A2_A2_B2

### Relational analysis result of NS_B2_A2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3718156, upper bound: 43.3674222
time: 0.79 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.33 seconds
NS_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -43.5605188, upper bound: 43.5922586
NS_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -43.5581766, upper bound: 43.5716737
NS_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -43.5811461, upper bound: 43.5747232
NS_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -43.5581766, upper bound: 43.5716737
NS_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -43.5569962, upper bound: 43.5769161
NS_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -43.5539466, upper bound: 43.5539466
NS_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -43.5737330, upper bound: 43.5782946
NS_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -43.5722960, upper bound: 43.5722960
NS_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -43.5604773, upper bound: 43.5721456
NS_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -43.5557536, upper bound: 43.5263719
NS_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -43.5224054, upper bound: 43.5670800
NS_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -43.5176817, upper bound: 43.5213063
NS_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -43.5442409, upper bound: 43.5553418
NS_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -43.5442409, upper bound: 43.5553418
NS_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -43.5442409, upper bound: 43.5553418
NS_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -43.5442409, upper bound: 43.5553418
NS_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -43.5721456, upper bound: 43.5604773
NS_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -43.5263719, upper bound: 43.5557536
NS_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -43.5721456, upper bound: 43.5604773
NS_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -43.5263719, upper bound: 43.5557536
NS_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -43.4317113, upper bound: 43.4934255
NS_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -43.4317113, upper bound: 43.4934255
NS_B2_A2_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 3.33
Output dim: 3, lower bound: -43.3538075, upper bound: 43.3432026
NS_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -43.3818490, upper bound: 43.3840848
NS_B2_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -43.4275308, upper bound: 43.4514684
NS_B2_A2_A2_A1_B2, status: Status.VERIFIED, split count: 5, time: 3.33
Output dim: 3, lower bound: -43.3798398, upper bound: 43.3575275
NS_B2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -43.4195067, upper bound: 43.4657565
NS_B2_A2_A2_A2_B2, status: Status.VERIFIED, split count: 5, time: 3.33
Output dim: 3, lower bound: -43.3718156, upper bound: 43.3674222

## BFS NS instance: NS_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -6.0080776, 22.7014885, -5.8062859, 21.9969387, -28.0050144, 28.5077724
1: -6.9569273, 25.7120094, -6.7116928, 24.9162731, -31.8731995, 32.4236984
2: -7.4386706, 25.7422943, -7.1873412, 24.9264069, -32.3650780, 32.9296341
3: -11.0225744, 26.3203468, -10.6498022, 25.4913673, -36.5139427, 36.9701462
4: -11.7293959, 25.3994999, -11.3617077, 24.5736980, -36.3030930, 36.7612076

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5759037, upper bound: 43.5759037
time: 0.62 seconds

## Relational analysis of NS_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5759037, upper bound: 43.5759037
time: 0.45 seconds

## BFS NS instance: NS_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -6.0080776, 22.7014885, -5.9676609, 22.6236305, -28.6317062, 28.6691475
1: -6.9569273, 25.7120094, -6.9145479, 25.6272087, -32.5841370, 32.6265526
2: -7.4386706, 25.7422943, -7.3910317, 25.6533833, -33.0920525, 33.1333275
3: -11.0225744, 26.3203468, -10.9741497, 26.2476044, -37.2701759, 37.2944908
4: -11.7293959, 25.3994999, -11.7002325, 25.2735844, -37.0029793, 37.0997314

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5759037, upper bound: 43.5759037
time: 0.46 seconds

## Relational analysis of NS_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5759037, upper bound: 43.5759037
time: 0.90 seconds

## BFS NS instance: NS_B1_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -6.3634715, 24.0477009, -6.0080776, 22.7014885, -29.0649605, 30.0557766
1: -7.3651905, 27.2656727, -6.9569273, 25.7120094, -33.0771942, 34.2225990
2: -7.8789954, 27.2677956, -7.4386706, 25.7422943, -33.6212883, 34.7064667
3: -11.6574259, 27.9324017, -11.0225744, 26.3203468, -37.9777679, 38.9549713
4: -12.4452915, 26.8532047, -11.7293959, 25.3994999, -37.8447914, 38.5825996

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_B1_A2_A1_B1

### Relational analysis result of NS_B1_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5581766, upper bound: 43.5716737
time: 0.50 seconds

## Relational analysis of NS_B1_A1_B1_A2_A1_B2

### Relational analysis result of NS_B1_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5581766, upper bound: 43.5716737
time: 0.45 seconds

## BFS NS instance: NS_B1_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -6.5370998, 24.7172279, -6.0080776, 22.7014885, -29.2385883, 30.7253036
1: -7.5843964, 28.0217705, -6.9569273, 25.7120094, -33.2964058, 34.9786987
2: -8.0978727, 28.0383968, -7.4386706, 25.7422943, -33.8401680, 35.4770660
3: -12.0083332, 28.7461624, -11.0225744, 26.3203468, -38.3286743, 39.7687378
4: -12.8104057, 27.6020432, -11.7293959, 25.3994999, -38.2098999, 39.3314400

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_B1_A2_A2_B1

### Relational analysis result of NS_B1_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5581766, upper bound: 43.5716737
time: 0.48 seconds

## Relational analysis of NS_B1_A1_B1_A2_A2_B2

### Relational analysis result of NS_B1_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5581766, upper bound: 43.5716737
time: 0.46 seconds

## BFS NS instance: NS_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -6.0080776, 22.7014885, -6.3634715, 24.0477009, -30.0557766, 29.0649605
1: -6.9569273, 25.7120094, -7.3651905, 27.2656727, -34.2225990, 33.0771942
2: -7.4386706, 25.7422943, -7.8789954, 27.2677956, -34.7064667, 33.6212883
3: -11.0225744, 26.3203468, -11.6574259, 27.9324017, -38.9549713, 37.9777679
4: -11.7293959, 25.3994999, -12.4452915, 26.8532047, -38.5825996, 37.8447914

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5716737, upper bound: 43.5581766
time: 0.46 seconds

## Relational analysis of NS_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5716737, upper bound: 43.5581766
time: 0.45 seconds

## BFS NS instance: NS_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -6.0080776, 22.7014885, -6.5370998, 24.7172279, -30.7253036, 29.2385883
1: -6.9569273, 25.7120094, -7.5843964, 28.0217705, -34.9786987, 33.2964058
2: -7.4386706, 25.7422943, -8.0978727, 28.0383968, -35.4770660, 33.8401680
3: -11.0225744, 26.3203468, -12.0083332, 28.7461624, -39.7687378, 38.3286781
4: -11.7293959, 25.3994999, -12.8104057, 27.6020432, -39.3314400, 38.2098999

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5716737, upper bound: 43.5581766
time: 0.43 seconds

## Relational analysis of NS_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5716737, upper bound: 43.5581766
time: 0.43 seconds

## BFS NS instance: NS_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -6.5748506, 24.7860813, -6.3499899, 24.0085678, -30.5834179, 31.1360703
1: -7.6228886, 28.1002808, -7.3510156, 27.2208977, -34.8437843, 35.4512978
2: -8.1419983, 28.1216526, -7.8660407, 27.2241821, -35.3661804, 35.9876938
3: -12.0498886, 28.8040485, -11.6391268, 27.8905048, -39.9403915, 40.4431725
4: -12.8321266, 27.7175007, -12.4185200, 26.8287506, -39.6608772, 40.1360207

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5701873, upper bound: 43.5553420
time: 0.44 seconds

## Relational analysis of NS_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5722960, upper bound: 43.5722960
time: 0.74 seconds

## Relational analysis of NS_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5722960, upper bound: 43.5722960
time: 0.98 seconds

## BFS NS instance: NS_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -6.5748506, 24.7860813, -6.4932985, 24.4664745, -31.0413246, 31.2793789
1: -7.6228886, 28.1002808, -7.5265026, 27.6794682, -35.3023567, 35.6267853
2: -8.1419983, 28.1216526, -8.0526819, 27.7724781, -35.9144745, 36.1743355
3: -12.0498886, 28.8040485, -11.9027405, 28.3914223, -40.4413109, 40.7067833
4: -12.8321266, 27.7175007, -12.6291609, 27.4867420, -40.3188591, 40.3466606

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5722960, upper bound: 43.5722960
time: 0.43 seconds

## Relational analysis of NS_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5722960, upper bound: 43.5722960
time: 0.53 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -8.6322088, 32.3104172, -5.0394983, 19.3376427, -27.9698524, 37.3499107
1: -10.1549702, 36.7054749, -5.7875409, 21.9354172, -32.0903816, 42.4930115
2: -10.6975584, 36.7274017, -6.2317314, 21.8455486, -32.5431061, 42.9591331
3: -15.9301043, 37.9201851, -9.2545710, 22.4035969, -38.3336983, 47.1747551
4: -16.8876266, 36.1860275, -10.0192795, 21.3928051, -38.2804337, 46.2053032

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A2_B1_A1_B1_B1

### Relational analysis result of NS_B1_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5604773, upper bound: 43.5721456
time: 0.44 seconds

## Relational analysis of NS_B1_A2_B1_A1_B1_B2

### Relational analysis result of NS_B1_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5604773, upper bound: 43.5721456
time: 0.43 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -8.6322088, 32.3104172, -5.5686293, 21.0664139, -29.6986237, 37.8790474
1: -10.1549702, 36.7054749, -6.4469433, 23.8585854, -34.0135460, 43.1524200
2: -10.6975584, 36.7274017, -6.8837023, 23.8715839, -34.5691376, 43.6110992
3: -15.9301043, 37.9201851, -10.2405462, 24.4116497, -40.3417549, 48.1607323
4: -16.8876266, 36.1860275, -10.9216919, 23.5361233, -40.4237480, 47.1077194

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A2_B1_A1_B2_B1

### Relational analysis result of NS_B1_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5557536, upper bound: 43.5263719
time: 0.46 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2_B2

### Relational analysis result of NS_B1_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5557536, upper bound: 43.5263719
time: 0.42 seconds

## BFS NS instance: NS_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.8174801, 33.0339127, -5.0394983, 19.3376427, -28.1551189, 38.0734024
1: -10.3915501, 37.5448494, -5.7875409, 21.9354172, -32.3269577, 43.3323860
2: -10.9315701, 37.5578690, -6.2317314, 21.8455486, -32.7771187, 43.7896004
3: -16.3067951, 38.8084564, -9.2545710, 22.4035969, -38.7103844, 48.0630264
4: -17.2786331, 36.9893837, -10.0192795, 21.3928051, -38.6714401, 47.0086632

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A2_B1_A2_B1_B1

### Relational analysis result of NS_B1_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5224054, upper bound: 43.5670800
time: 0.48 seconds

## Relational analysis of NS_B1_A2_B1_A2_B1_B2

### Relational analysis result of NS_B1_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5224054, upper bound: 43.5670800
time: 0.43 seconds

## BFS NS instance: NS_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -8.8174801, 33.0339127, -5.5686293, 21.0664139, -29.8838902, 38.6025429
1: -10.3915501, 37.5448494, -6.4469433, 23.8585854, -34.2501297, 43.9917908
2: -10.9315701, 37.5578690, -6.8837023, 23.8715839, -34.8031502, 44.4415703
3: -16.3067951, 38.8084564, -10.2405462, 24.4116497, -40.7184448, 49.0490036
4: -17.2786331, 36.9893837, -10.9216919, 23.5361233, -40.8147583, 47.9110756

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A2_B1_A2_B2_B1

### Relational analysis result of NS_B1_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5176817, upper bound: 43.5213063
time: 0.46 seconds

## Relational analysis of NS_B1_A2_B1_A2_B2_B2

### Relational analysis result of NS_B1_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5176817, upper bound: 43.5213063
time: 0.49 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -8.6322088, 32.3104172, -6.5283041, 24.7212219, -33.3534317, 38.8387222
1: -10.1549702, 36.7054749, -7.5771909, 28.0262680, -38.1812363, 44.2826653
2: -10.6975584, 36.7274017, -8.0923138, 28.0525475, -38.7501068, 44.8197174
3: -15.9301043, 37.9201851, -11.9928169, 28.7598476, -44.6899490, 49.9130020
4: -16.8876266, 36.1860275, -12.8025274, 27.6119270, -44.4995499, 48.9885559

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B2_A1_B1_B1

### Relational analysis result of NS_B1_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5570367, upper bound: 43.5582296
time: 0.45 seconds

## Relational analysis of NS_B1_A2_B2_A1_B1_B2

### Relational analysis result of NS_B1_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5546630, upper bound: 43.5241353
time: 0.46 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -8.6322088, 32.3104172, -6.7093639, 25.4152355, -34.0474434, 39.0197792
1: -10.1549702, 36.7054749, -7.8063669, 28.8139172, -38.9688873, 44.5118370
2: -10.6975584, 36.7274017, -8.3207026, 28.8522396, -39.5497971, 45.0481033
3: -15.9301043, 37.9201851, -12.3585405, 29.6142330, -45.5443382, 50.2787247
4: -16.8876266, 36.1860275, -13.1818914, 28.3913708, -45.2789993, 49.3679161

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B2_A1_B2_B1

### Relational analysis result of NS_B1_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5570367, upper bound: 43.5582296
time: 0.72 seconds

## Relational analysis of NS_B1_A2_B2_A1_B2_B2

### Relational analysis result of NS_B1_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5546630, upper bound: 43.5241353
time: 0.44 seconds

## BFS NS instance: NS_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -8.8174801, 33.0339127, -6.5283041, 24.7212219, -33.5387039, 39.5622177
1: -10.3915501, 37.5448494, -7.5771909, 28.0262680, -38.4178162, 45.1220398
2: -10.9315701, 37.5578690, -8.0923138, 28.0525475, -38.9841156, 45.6501846
3: -16.3067951, 38.8084564, -11.9928169, 28.7598476, -45.0666389, 50.8012733
4: -17.2786331, 36.9893837, -12.8025274, 27.6119270, -44.8905602, 49.7919121

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B2_A2_B1_B1

### Relational analysis result of NS_B1_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5189648, upper bound: 43.5531641
time: 0.44 seconds

## Relational analysis of NS_B1_A2_B2_A2_B1_B2

### Relational analysis result of NS_B1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5165910, upper bound: 43.5190697
time: 0.55 seconds

## BFS NS instance: NS_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -8.8174801, 33.0339127, -6.7093639, 25.4152355, -34.2327156, 39.7432747
1: -10.3915501, 37.5448494, -7.8063669, 28.8139172, -39.2054672, 45.3512115
2: -10.9315701, 37.5578690, -8.3207026, 28.8522396, -39.7838097, 45.8785706
3: -16.3067951, 38.8084564, -12.3585405, 29.6142330, -45.9210281, 51.1669960
4: -17.2786331, 36.9893837, -13.1818914, 28.3913708, -45.6700058, 50.1712761

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B2_A2_B2_B1

### Relational analysis result of NS_B1_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5189648, upper bound: 43.5531641
time: 0.55 seconds

## Relational analysis of NS_B1_A2_B2_A2_B2_B2

### Relational analysis result of NS_B1_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5165910, upper bound: 43.5190697
time: 0.57 seconds

## BFS NS instance: NS_B2_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -5.9647384, 22.4902878, -8.6322088, 32.3104172, -38.2751541, 31.1224976
1: -6.9048386, 25.4210529, -10.1549702, 36.7054749, -43.6103134, 35.5760193
2: -7.3848820, 25.5047226, -10.6975584, 36.7274017, -44.1122818, 36.2022705
3: -10.9337454, 26.0330429, -15.9301043, 37.9201851, -48.8539276, 41.9631462
4: -11.5946884, 25.2559605, -16.8876266, 36.1860275, -47.7807121, 42.1435852

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B2_A1_B1_A1_A1_B1

### Relational analysis result of NS_B2_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5890489, upper bound: 43.5617729
time: 0.44 seconds

## Relational analysis of NS_B2_A1_B1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_B1_A1_A1_B1

### Relational analysis result of NS_B2_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5596421, upper bound: 43.5581364
time: 0.47 seconds

## Relational analysis of NS_B2_A1_B1_A1_A1_B2

### Relational analysis result of NS_B2_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5596421, upper bound: 43.5581364
time: 0.67 seconds

## BFS NS instance: NS_B2_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -6.4972239, 24.2319641, -8.6322088, 32.3104172, -38.8076401, 32.8641624
1: -7.5682106, 27.3599796, -10.1549702, 36.7054749, -44.2736855, 37.5149460
2: -8.0439062, 27.5397072, -10.6975584, 36.7274017, -44.7713013, 38.2372665
3: -11.9258957, 28.0709877, -15.9301043, 37.9201851, -49.8460770, 44.0010910
4: -12.5062847, 27.4008636, -16.8876266, 36.1860275, -48.6923141, 44.2884903

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_B1_A1_A2_A1

### Relational analysis result of NS_B2_A1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5414172, upper bound: 43.5329154
time: 0.62 seconds

## Relational analysis of NS_B2_A1_B1_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B2_A1_B1_A1_A2_B1

### Relational analysis result of NS_B2_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5229003, upper bound: 43.5535249
time: 0.49 seconds

## Relational analysis of NS_B2_A1_B1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_B1_A1_A2_B1

### Relational analysis result of NS_B2_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5596421, upper bound: 43.5581364
time: 0.64 seconds

## Relational analysis of NS_B2_A1_B1_A1_A2_B2

### Relational analysis result of NS_B2_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5596421, upper bound: 43.5581364
time: 0.61 seconds

## BFS NS instance: NS_B2_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -6.1316075, 23.1411781, -8.6322088, 32.3104172, -38.4420204, 31.7733879
1: -7.1131840, 26.1596413, -10.1549702, 36.7054749, -43.8186569, 36.3146095
2: -7.5950131, 26.2545033, -10.6975584, 36.7274017, -44.3224144, 36.9520569
3: -11.2687273, 26.8291836, -15.9301043, 37.9201851, -49.1889114, 42.7592850
4: -11.9468098, 25.9775772, -16.8876266, 36.1860275, -48.1328354, 42.8652039

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_B1_A2_A1_B1

### Relational analysis result of NS_B2_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5263719, upper bound: 43.5557536
time: 0.72 seconds

## Relational analysis of NS_B2_A1_B1_A2_A1_B2

### Relational analysis result of NS_B2_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5263719, upper bound: 43.5557536
time: 0.47 seconds

## BFS NS instance: NS_B2_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -6.6653934, 24.8864861, -8.6322088, 32.3104172, -38.9758072, 33.5186844
1: -7.7828097, 28.1194229, -10.1549702, 36.7054749, -44.4882774, 38.2743912
2: -8.2539454, 28.2906685, -10.6975584, 36.7274017, -44.9813461, 38.9882278
3: -12.2706099, 28.8829918, -15.9301043, 37.9201851, -50.1907921, 44.8130951
4: -12.8631258, 28.1254616, -16.8876266, 36.1860275, -49.0491524, 45.0130882

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_B1_A2_A2_A1

### Relational analysis result of NS_B2_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5080760, upper bound: 43.5305302
time: 0.62 seconds

## Relational analysis of NS_B2_A1_B1_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_B1_A2_A2_B1

### Relational analysis result of NS_B2_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5263719, upper bound: 43.5557536
time: 0.46 seconds

## Relational analysis of NS_B2_A1_B1_A2_A2_B2

### Relational analysis result of NS_B2_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5263719, upper bound: 43.5557536
time: 0.47 seconds

## BFS NS instance: NS_B2_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -8.6322088, 32.3104172, -8.5925446, 32.1775818, -40.8097839, 40.9029617
1: -10.1549702, 36.7054749, -10.1066513, 36.5556908, -46.7106590, 46.8121262
2: -10.6975584, 36.7274017, -10.6485958, 36.5736465, -47.2712021, 47.3759995
3: -15.9301043, 37.9201851, -15.8576345, 37.7638512, -53.6939545, 53.7778206
4: -16.8876266, 36.1860275, -16.8185234, 36.0260582, -52.9136848, 53.0045433

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B2_A2_A1_B1_B1_A1

### Relational analysis result of NS_B2_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4196544, upper bound: 43.4523578
time: 0.51 seconds

## Relational analysis of NS_B2_A2_A1_B1_B1_A2

### Relational analysis result of NS_B2_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4303320, upper bound: 43.4760308
time: 0.51 seconds

## BFS NS instance: NS_B2_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -8.6322088, 32.3104172, -8.7776985, 32.9010162, -41.5332222, 41.0881119
1: -10.1549702, 36.7054749, -10.3430977, 37.3951607, -47.5501251, 47.0485725
2: -10.6975584, 36.7274017, -10.8825235, 37.4039421, -48.1014977, 47.6099205
3: -15.9301043, 37.9201851, -16.2341270, 38.6523247, -54.5824280, 54.1543121
4: -16.8876266, 36.1860275, -17.2096958, 36.8289986, -53.7166252, 53.3957100

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_A1_B1_B2_A1

### Relational analysis result of NS_B2_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4277048, upper bound: 43.4514723
time: 0.70 seconds

## Relational analysis of NS_B2_A2_A1_B1_B2_A2

### Relational analysis result of NS_B2_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4256336, upper bound: 43.4678438
time: 0.53 seconds

## BFS NS instance: NS_B2_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -8.6322088, 32.3104172, -9.3598814, 35.1582031, -43.7904091, 41.6702995
1: -10.1549702, 36.7054749, -11.0692644, 40.0464897, -50.1704407, 47.7747383
2: -10.6975584, 36.7274017, -11.6192999, 39.9427032, -50.6402626, 48.3467026
3: -15.9301043, 37.9201851, -17.3481617, 41.4529648, -57.3366737, 55.2683449
4: -16.8876266, 36.1860275, -18.4878654, 39.2475166, -56.1351433, 54.6738930

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B2_A2_A1_B2_B2_A1

### Relational analysis result of NS_B2_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3818490, upper bound: 43.3840848
time: 0.95 seconds

## Relational analysis of NS_B2_A2_A1_B2_B2_A2

### Relational analysis result of NS_B2_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3818490, upper bound: 43.3840848
time: 0.56 seconds

## BFS NS instance: NS_B2_A2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -8.5439262, 32.0631142, -8.8156204, 32.9617119, -41.5056381, 40.8787308
1: -10.0602074, 36.4324837, -10.3809900, 37.4477654, -47.5079727, 46.8134727
2: -10.5933409, 36.4442482, -10.9258604, 37.4781151, -48.0714531, 47.3700981
3: -15.7987118, 37.6562881, -16.2741280, 38.6946754, -54.4933853, 53.9304161
4: -16.7670708, 35.8889656, -17.2310925, 36.9354782, -53.7025490, 53.1200562

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B2_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B2_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_A2_A1_B1_B1

### Relational analysis result of NS_B2_A2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4275308, upper bound: 43.4514301
time: 0.85 seconds

## Relational analysis of NS_B2_A2_A2_A1_B1_B2

### Relational analysis result of NS_B2_A2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4071380, upper bound: 43.4043641
time: 0.49 seconds

## BFS NS instance: NS_B2_A2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -9.1115875, 33.9125404, -8.8156204, 32.9617119, -42.0732994, 42.7281609
1: -10.7727814, 38.5158424, -10.3809900, 37.4477654, -48.2205467, 48.8968315
2: -11.2928276, 38.6079216, -10.9258604, 37.4781151, -48.7709427, 49.5337753
3: -16.8676147, 39.8401985, -16.2741280, 38.6946754, -55.5622902, 56.1143265
4: -17.7444897, 38.1677361, -17.2310925, 36.9354782, -54.6799698, 55.3988266

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B2_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B2_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_A2_A2_B1_A1

### Relational analysis result of NS_B2_A2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3667007, upper bound: 43.4186522
time: 0.65 seconds

## Relational analysis of NS_B2_A2_A2_A2_B1_A2

### Relational analysis result of NS_B2_A2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3991139, upper bound: 43.4184521
time: 0.60 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.86 seconds
NS_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5759037, upper bound: 43.5759037
NS_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5759037, upper bound: 43.5759037
NS_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5759037, upper bound: 43.5759037
NS_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5759037, upper bound: 43.5759037
NS_B1_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5581766, upper bound: 43.5716737
NS_B1_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5581766, upper bound: 43.5716737
NS_B1_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5581766, upper bound: 43.5716737
NS_B1_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5581766, upper bound: 43.5716737
NS_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5716737, upper bound: 43.5581766
NS_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5716737, upper bound: 43.5581766
NS_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5716737, upper bound: 43.5581766
NS_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5716737, upper bound: 43.5581766
NS_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5722960, upper bound: 43.5722960
NS_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5722960, upper bound: 43.5722960
NS_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5722960, upper bound: 43.5722960
NS_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5722960, upper bound: 43.5722960
NS_B1_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5604773, upper bound: 43.5721456
NS_B1_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5604773, upper bound: 43.5721456
NS_B1_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5557536, upper bound: 43.5263719
NS_B1_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5557536, upper bound: 43.5263719
NS_B1_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5224054, upper bound: 43.5670800
NS_B1_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5224054, upper bound: 43.5670800
NS_B1_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5176817, upper bound: 43.5213063
NS_B1_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5176817, upper bound: 43.5213063
NS_B1_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5570367, upper bound: 43.5582296
NS_B1_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5546630, upper bound: 43.5241353
NS_B1_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5570367, upper bound: 43.5582296
NS_B1_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5546630, upper bound: 43.5241353
NS_B1_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5189648, upper bound: 43.5531641
NS_B1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5165910, upper bound: 43.5190697
NS_B1_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5189648, upper bound: 43.5531641
NS_B1_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5165910, upper bound: 43.5190697
NS_B2_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5596421, upper bound: 43.5581364
NS_B2_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5596421, upper bound: 43.5581364
NS_B2_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5596421, upper bound: 43.5581364
NS_B2_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5596421, upper bound: 43.5581364
NS_B2_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5263719, upper bound: 43.5557536
NS_B2_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5263719, upper bound: 43.5557536
NS_B2_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5263719, upper bound: 43.5557536
NS_B2_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.5263719, upper bound: 43.5557536
NS_B2_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.4196544, upper bound: 43.4523578
NS_B2_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.4303320, upper bound: 43.4760308
NS_B2_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.4277048, upper bound: 43.4514723
NS_B2_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.4256336, upper bound: 43.4678438
NS_B2_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.3818490, upper bound: 43.3840848
NS_B2_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.3818490, upper bound: 43.3840848
NS_B2_A2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.4275308, upper bound: 43.4514301
NS_B2_A2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.4071380, upper bound: 43.4043641
NS_B2_A2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.3667007, upper bound: 43.4186522
NS_B2_A2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 3, lower bound: -43.3991139, upper bound: 43.4184521

## BFS NS instance: NS_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -5.8062859, 21.9969387, -5.8062859, 21.9969387, -27.8032227, 27.8032227
1: -6.7116928, 24.9162731, -6.7116928, 24.9162731, -31.6279640, 31.6279659
2: -7.1873412, 24.9264069, -7.1873412, 24.9264069, -32.1137466, 32.1137466
3: -10.6498022, 25.4913673, -10.6498022, 25.4913673, -36.1411705, 36.1411705
4: -11.3617077, 24.5736980, -11.3617077, 24.5736980, -35.9354057, 35.9354057

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B1_A1_B1_A1_A1

### Relational analysis result of NS_B1_A1_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5735628, upper bound: 43.5624854
time: 0.47 seconds

## Relational analysis of NS_B1_A1_B1_A1_B1_A1_A2

### Relational analysis result of NS_B1_A1_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5274674, upper bound: 43.5577617
time: 0.43 seconds

## BFS NS instance: NS_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -5.9676609, 22.6236305, -5.8062859, 21.9969387, -27.9645977, 28.4299164
1: -6.9145479, 25.6272087, -6.7116928, 24.9162731, -31.8308220, 32.3388977
2: -7.3910317, 25.6533833, -7.1873412, 24.9264069, -32.3174400, 32.8407249
3: -10.9741497, 26.2476044, -10.6498022, 25.4913673, -36.4655151, 36.8974075
4: -11.7002325, 25.2735844, -11.3617077, 24.5736980, -36.2739296, 36.6352921

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5321373, upper bound: 43.5914800
time: 0.49 seconds

## Relational analysis of NS_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5274674, upper bound: 43.5577617
time: 0.45 seconds

## BFS NS instance: NS_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -5.8062859, 21.9969387, -5.9676609, 22.6236305, -28.4299164, 27.9645977
1: -6.7116928, 24.9162731, -6.9145479, 25.6272087, -32.3388977, 31.8308220
2: -7.1873412, 24.9264069, -7.3910317, 25.6533833, -32.8407249, 32.3174362
3: -10.6498022, 25.4913673, -10.9741497, 26.2476044, -36.8974075, 36.4655151
4: -11.3617077, 24.5736980, -11.7002325, 25.2735844, -36.6352921, 36.2739296

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B1_A1_B2_A1_A1

### Relational analysis result of NS_B1_A1_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5711800, upper bound: 43.5298083
time: 0.63 seconds

## Relational analysis of NS_B1_A1_B1_A1_B2_A1_A2

### Relational analysis result of NS_B1_A1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5250846, upper bound: 43.5250846
time: 0.43 seconds

## BFS NS instance: NS_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -5.9676609, 22.6236305, -5.9676609, 22.6236305, -28.5912895, 28.5912895
1: -6.9145479, 25.6272087, -6.9145479, 25.6272087, -32.5417519, 32.5417480
2: -7.3910317, 25.6533833, -7.3910317, 25.6533833, -33.0444145, 33.0444145
3: -10.9741497, 26.2476044, -10.9741497, 26.2476044, -37.2217560, 37.2217522
4: -11.7002325, 25.2735844, -11.7002325, 25.2735844, -36.9738083, 36.9738083

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B1_A1_B2_A2_A1

### Relational analysis result of NS_B1_A1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5711800, upper bound: 43.5298083
time: 0.47 seconds

## Relational analysis of NS_B1_A1_B1_A1_B2_A2_A2

### Relational analysis result of NS_B1_A1_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5250846, upper bound: 43.5250846
time: 0.49 seconds

## BFS NS instance: NS_B1_A1_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -6.3634715, 24.0477009, -5.8062859, 21.9969387, -28.3604107, 29.8539867
1: -7.3651905, 27.2656727, -6.7116928, 24.9162731, -32.2814636, 33.9773636
2: -7.8789954, 27.2677956, -7.1873412, 24.9264069, -32.8054008, 34.4551353
3: -11.6574259, 27.9324017, -10.6498022, 25.4913673, -37.1487923, 38.5822029
4: -12.4452915, 26.8532047, -11.3617077, 24.5736980, -37.0189896, 38.2149124

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B1_A2_A1_B1_B1

### Relational analysis result of NS_B1_A1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5624102, upper bound: 43.5735321
time: 0.51 seconds

## Relational analysis of NS_B1_A1_B1_A2_A1_B1_B2

### Relational analysis result of NS_B1_A1_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5576866, upper bound: 43.5274367
time: 0.48 seconds

## BFS NS instance: NS_B1_A1_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -6.3634715, 24.0477009, -5.9676609, 22.6236305, -28.9871025, 30.0153599
1: -7.3651905, 27.2656727, -6.9145479, 25.6272087, -32.9923935, 34.1802177
2: -7.8789954, 27.2677956, -7.3910317, 25.6533833, -33.5323753, 34.6588287
3: -11.6574259, 27.9324017, -10.9741497, 26.2476044, -37.9050293, 38.9065514
4: -12.4452915, 26.8532047, -11.7002325, 25.2735844, -37.7188721, 38.5534363

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B1_A2_A1_B2_B1

### Relational analysis result of NS_B1_A1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5624102, upper bound: 43.5735321
time: 0.45 seconds

## Relational analysis of NS_B1_A1_B1_A2_A1_B2_B2

### Relational analysis result of NS_B1_A1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5576866, upper bound: 43.5274367
time: 0.69 seconds

## BFS NS instance: NS_B1_A1_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -6.5370998, 24.7172279, -5.8062859, 21.9969387, -28.5340385, 30.5235138
1: -7.5843964, 28.0217705, -6.7116928, 24.9162731, -32.5006714, 34.7334633
2: -8.0978727, 28.0383968, -7.1873412, 24.9264069, -33.0242805, 35.2257385
3: -12.0083332, 28.7461624, -10.6498022, 25.4913673, -37.4997025, 39.3959656
4: -12.8104057, 27.6020432, -11.3617077, 24.5736980, -37.3841019, 38.9637527

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B1_A2_A2_B1_B1

### Relational analysis result of NS_B1_A1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5273146, upper bound: 43.5698741
time: 0.45 seconds

## Relational analysis of NS_B1_A1_B1_A2_A2_B1_B2

### Relational analysis result of NS_B1_A1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5225909, upper bound: 43.5237787
time: 0.47 seconds

## BFS NS instance: NS_B1_A1_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -6.5370998, 24.7172279, -5.9676609, 22.6236305, -29.1607304, 30.6848869
1: -7.5843964, 28.0217705, -6.9145479, 25.6272087, -33.2116051, 34.9363136
2: -8.0978727, 28.0383968, -7.3910317, 25.6533833, -33.7512512, 35.4294281
3: -12.0083332, 28.7461624, -10.9741497, 26.2476044, -38.2559357, 39.7203140
4: -12.8104057, 27.6020432, -11.7002325, 25.2735844, -38.0839844, 39.3022728

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B1_A2_A2_B2_A1

### Relational analysis result of NS_B1_A1_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5534529, upper bound: 43.5255783
time: 0.66 seconds

## Relational analysis of NS_B1_A1_B1_A2_A2_B2_A2

### Relational analysis result of NS_B1_A1_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5225909, upper bound: 43.5237787
time: 0.51 seconds

## BFS NS instance: NS_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -5.8062859, 21.9969387, -6.3634715, 24.0477009, -29.8539848, 28.3604107
1: -6.7116928, 24.9162731, -7.3651905, 27.2656727, -33.9773636, 32.2814636
2: -7.1873412, 24.9264069, -7.8789954, 27.2677956, -34.4551353, 32.8054008
3: -10.6498022, 25.4913673, -11.6574259, 27.9324017, -38.5822029, 37.1487923
4: -11.3617077, 24.5736980, -12.4452915, 26.8532047, -38.2149124, 37.0189896

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B2_A1_B1_A1_A1

### Relational analysis result of NS_B1_A1_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5735321, upper bound: 43.5624102
time: 0.47 seconds

## Relational analysis of NS_B1_A1_B2_A1_B1_A1_A2

### Relational analysis result of NS_B1_A1_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5274367, upper bound: 43.5576866
time: 1.67 seconds

## BFS NS instance: NS_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -5.9676609, 22.6236305, -6.3634715, 24.0477009, -30.0153599, 28.9871025
1: -6.9145479, 25.6272087, -7.3651905, 27.2656727, -34.1802177, 32.9923973
2: -7.3910317, 25.6533833, -7.8789954, 27.2677956, -34.6588287, 33.5323715
3: -10.9741497, 26.2476044, -11.6574259, 27.9324017, -38.9065514, 37.9050293
4: -11.7002325, 25.2735844, -12.4452915, 26.8532047, -38.5534325, 37.7188721

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B2_A1_B1_A2_A1

### Relational analysis result of NS_B1_A1_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5735321, upper bound: 43.5624102
time: 0.48 seconds

## Relational analysis of NS_B1_A1_B2_A1_B1_A2_A2

### Relational analysis result of NS_B1_A1_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5274367, upper bound: 43.5576866
time: 0.85 seconds

## BFS NS instance: NS_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -5.8062859, 21.9969387, -6.5370998, 24.7172279, -30.5235138, 28.5340385
1: -6.7116928, 24.9162731, -7.5843964, 28.0217705, -34.7334633, 32.5006714
2: -7.1873412, 24.9264069, -8.0978727, 28.0383968, -35.2257385, 33.0242805
3: -10.6498022, 25.4913673, -12.0083332, 28.7461624, -39.3959656, 37.4997025
4: -11.3617077, 24.5736980, -12.8104057, 27.6020432, -38.9637527, 37.3841019

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B2_A1_B2_A1_A1

### Relational analysis result of NS_B1_A1_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5698741, upper bound: 43.5273146
time: 0.71 seconds

## Relational analysis of NS_B1_A1_B2_A1_B2_A1_A2

### Relational analysis result of NS_B1_A1_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5237787, upper bound: 43.5225909
time: 0.46 seconds

## BFS NS instance: NS_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -5.9676609, 22.6236305, -6.5370998, 24.7172279, -30.6848869, 29.1607304
1: -6.9145479, 25.6272087, -7.5843964, 28.0217705, -34.9363136, 33.2116051
2: -7.3910317, 25.6533833, -8.0978727, 28.0383968, -35.4294281, 33.7512512
3: -10.9741497, 26.2476044, -12.0083332, 28.7461624, -39.7203140, 38.2559357
4: -11.7002325, 25.2735844, -12.8104057, 27.6020432, -39.3022728, 38.0839844

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5255783, upper bound: 43.5534529
time: 0.44 seconds

## Relational analysis of NS_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5237787, upper bound: 43.5225909
time: 0.47 seconds

## BFS NS instance: NS_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -6.3499899, 24.0085678, -6.3499899, 24.0085678, -30.3585587, 30.3585587
1: -7.3510156, 27.2208977, -7.3510156, 27.2208977, -34.5719032, 34.5719032
2: -7.8660407, 27.2241821, -7.8660407, 27.2241821, -35.0902214, 35.0902214
3: -11.6391268, 27.8905048, -11.6391268, 27.8905048, -39.5296326, 39.5296326
4: -12.4185200, 26.8287506, -12.4185200, 26.8287506, -39.2472687, 39.2472687

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_B2_A2_B1_A1_A1

### Relational analysis result of NS_B1_A1_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5694878, upper bound: 43.5306737
time: 0.44 seconds

## Relational analysis of NS_B1_A1_B2_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## BFS NS instance: NS_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -6.4932985, 24.4664745, -6.3499899, 24.0085678, -30.5018654, 30.8164635
1: -7.5265026, 27.6794682, -7.3510156, 27.2208977, -34.7473907, 35.0304794
2: -8.0526819, 27.7724781, -7.8660407, 27.2241821, -35.2768631, 35.6385155
3: -11.9027405, 28.3914223, -11.6391268, 27.8905048, -39.7932434, 40.0305481
4: -12.6291609, 27.4867420, -12.4185200, 26.8287506, -39.4579124, 39.9052544

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_B2_A2_B1_A2_A1

### Relational analysis result of NS_B1_A1_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5694878, upper bound: 43.5306737
time: 0.79 seconds

## Relational analysis of NS_B1_A1_B2_A2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## BFS NS instance: NS_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -6.3499899, 24.0085678, -6.4932985, 24.4664745, -30.8164635, 30.5018654
1: -7.3510156, 27.2208977, -7.5265026, 27.6794682, -35.0304794, 34.7473907
2: -7.8660407, 27.2241821, -8.0526819, 27.7724781, -35.6385155, 35.2768631
3: -11.6391268, 27.8905048, -11.9027405, 28.3914223, -40.0305481, 39.7932434
4: -12.4185200, 26.8287506, -12.6291609, 27.4867420, -39.9052544, 39.4579124

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## BFS NS instance: NS_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6.4932985, 24.4664745, -6.4932985, 24.4664745, -30.9597721, 30.9597740
1: -7.5265026, 27.6794682, -7.5265026, 27.6794682, -35.2059669, 35.2059669
2: -8.0526819, 27.7724781, -8.0526819, 27.7724781, -35.8251610, 35.8251610
3: -11.9027405, 28.3914223, -11.9027405, 28.3914223, -40.2941589, 40.2941589
4: -12.6291609, 27.4867420, -12.6291609, 27.4867420, -40.1158981, 40.1158981

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## BFS NS instance: NS_B1_A2_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -8.6322088, 32.3104172, -4.8395281, 18.6394978, -27.2717056, 37.1499443
1: -10.1549702, 36.7054749, -5.5451407, 21.1490383, -31.3040047, 42.2506142
2: -10.6975584, 36.7274017, -5.9836831, 21.0349770, -31.7325344, 42.7110863
3: -15.9301043, 37.9201851, -8.8850317, 21.5857143, -37.5158195, 46.8052177
4: -16.8876266, 36.1860275, -9.6547794, 20.5678463, -37.4554749, 45.8408012

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_A1

### Relational analysis result of NS_B1_A2_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5604773, upper bound: 43.5721456
time: 0.46 seconds

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_A2

### Relational analysis result of NS_B1_A2_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5604773, upper bound: 43.5721456
time: 0.43 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -8.6322088, 32.3104172, -5.0165229, 19.3200130, -27.9522209, 37.3269348
1: -10.1549702, 36.7054749, -5.7698326, 21.9297009, -32.0846634, 42.4753075
2: -10.6975584, 36.7274017, -6.2069774, 21.8248901, -32.5224457, 42.9343796
3: -15.9301043, 37.9201851, -9.2441177, 22.4114609, -38.3415642, 47.1643028
4: -16.8876266, 36.1860275, -10.0225515, 21.3482018, -38.2358246, 46.2085800

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A1

### Relational analysis result of NS_B1_A2_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5604773, upper bound: 43.5721456
time: 0.48 seconds

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A2

### Relational analysis result of NS_B1_A2_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5604773, upper bound: 43.5721456
time: 0.44 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -8.6322088, 32.3104172, -5.3725829, 20.3826561, -29.0148659, 37.6829987
1: -10.1549702, 36.7054749, -6.2075019, 23.0882111, -33.2431755, 42.9129753
2: -10.6975584, 36.7274017, -6.6401882, 23.0763321, -33.7738914, 43.3675919
3: -15.9301043, 37.9201851, -9.8753433, 23.6089172, -39.5390205, 47.7955284
4: -16.8876266, 36.1860275, -10.5646782, 22.7292194, -39.6168442, 46.7507057

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B1_A1_B2_B1_A1

### Relational analysis result of NS_B1_A2_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5557536, upper bound: 43.5263719
time: 0.51 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2_B1_A2

### Relational analysis result of NS_B1_A2_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5557536, upper bound: 43.5263719
time: 0.47 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -8.6322088, 32.3104172, -5.4863067, 20.8377094, -29.4699173, 37.7967224
1: -10.1549702, 36.7054749, -6.3514509, 23.6165676, -33.7715302, 43.0569267
2: -10.6975584, 36.7274017, -6.7845912, 23.6114349, -34.3089943, 43.5119934
3: -15.9301043, 37.9201851, -10.1113768, 24.1738873, -40.1039925, 48.0315628
4: -16.8876266, 36.1860275, -10.8178120, 23.2331581, -40.1207848, 47.0038376

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A2_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A2_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B1_A1_B2_B2_A1

### Relational analysis result of NS_B1_A2_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5557536, upper bound: 43.5263719
time: 0.46 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2_B2_A2

### Relational analysis result of NS_B1_A2_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5557536, upper bound: 43.5263719
time: 0.42 seconds

## BFS NS instance: NS_B1_A2_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -8.8174801, 33.0339127, -4.8395281, 18.6394978, -27.4569740, 37.8734398
1: -10.3915501, 37.5448494, -5.5451407, 21.1490383, -31.5405846, 43.0899887
2: -10.9315701, 37.5578690, -5.9836831, 21.0349770, -31.9665470, 43.5415535
3: -16.3067951, 38.8084564, -8.8850317, 21.5857143, -37.8925095, 47.6934891
4: -17.2786331, 36.9893837, -9.6547794, 20.5678463, -37.8464813, 46.6441650

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B1_A2_B1_B1_A1

### Relational analysis result of NS_B1_A2_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5224054, upper bound: 43.5670800
time: 0.68 seconds

## Relational analysis of NS_B1_A2_B1_A2_B1_B1_A2

### Relational analysis result of NS_B1_A2_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5224054, upper bound: 43.5670800
time: 0.49 seconds

## BFS NS instance: NS_B1_A2_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -8.8174801, 33.0339127, -5.0165229, 19.3200130, -28.1374893, 38.0504303
1: -10.3915501, 37.5448494, -5.7698326, 21.9297009, -32.3212433, 43.3146820
2: -10.9315701, 37.5578690, -6.2069774, 21.8248901, -32.7564621, 43.7648468
3: -16.3067951, 38.8084564, -9.2441177, 22.4114609, -38.7182541, 48.0525742
4: -17.2786331, 36.9893837, -10.0225515, 21.3482018, -38.6268311, 47.0119362

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B1_A2_B1_B2_A1

### Relational analysis result of NS_B1_A2_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5224054, upper bound: 43.5670800
time: 0.70 seconds

## Relational analysis of NS_B1_A2_B1_A2_B1_B2_A2

### Relational analysis result of NS_B1_A2_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5224054, upper bound: 43.5670800
time: 0.44 seconds

## BFS NS instance: NS_B1_A2_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -8.8174801, 33.0339127, -5.3725829, 20.3826561, -29.2001324, 38.4064941
1: -10.3915501, 37.5448494, -6.2075019, 23.0882111, -33.4797516, 43.7523499
2: -10.9315701, 37.5578690, -6.6401882, 23.0763321, -34.0079041, 44.1980591
3: -16.3067951, 38.8084564, -9.8753433, 23.6089172, -39.9157104, 48.6837997
4: -17.2786331, 36.9893837, -10.5646782, 22.7292194, -40.0078506, 47.5540619

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A2_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A2_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B1_A2_B2_B1_A1

### Relational analysis result of NS_B1_A2_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5176817, upper bound: 43.5213063
time: 0.52 seconds

## Relational analysis of NS_B1_A2_B1_A2_B2_B1_A2

### Relational analysis result of NS_B1_A2_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5176817, upper bound: 43.5213063
time: 0.43 seconds

## BFS NS instance: NS_B1_A2_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -8.8174801, 33.0339127, -5.4863067, 20.8377094, -29.6551857, 38.5202179
1: -10.3915501, 37.5448494, -6.3514509, 23.6165676, -34.0081100, 43.8963013
2: -10.9315701, 37.5578690, -6.7845912, 23.6114349, -34.5430069, 44.3424606
3: -16.3067951, 38.8084564, -10.1113768, 24.1738873, -40.4806786, 48.9198341
4: -17.2786331, 36.9893837, -10.8178120, 23.2331581, -40.5117912, 47.8071976

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B1_A2_B2_B2_A1

### Relational analysis result of NS_B1_A2_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5176817, upper bound: 43.5213063
time: 0.48 seconds

## Relational analysis of NS_B1_A2_B1_A2_B2_B2_A2

### Relational analysis result of NS_B1_A2_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5176817, upper bound: 43.5213063
time: 0.43 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -8.6322088, 32.3104172, -6.2620754, 23.7622166, -32.3944168, 38.5724945
1: -10.1549702, 36.7054749, -7.2577133, 26.9220257, -37.0769958, 43.9631844
2: -10.6975584, 36.7274017, -7.7621226, 26.9545021, -37.6520615, 44.4895248
3: -15.9301043, 37.9201851, -11.5024481, 27.6255684, -43.5556717, 49.4226303
4: -16.8876266, 36.1860275, -12.3027077, 26.5307827, -43.4184113, 48.4887314

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A1

### Relational analysis result of NS_B1_A2_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5581364, upper bound: 43.5596421
time: 0.73 seconds

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2

### Relational analysis result of NS_B1_A2_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5581364, upper bound: 43.5596421
time: 0.51 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -8.6322088, 32.3104172, -6.8138876, 25.5695591, -34.2017593, 39.1242943
1: -10.1549702, 36.7054749, -7.9509349, 28.9397583, -39.0947151, 44.6564102
2: -10.6975584, 36.7274017, -8.4420090, 29.0706272, -39.7681808, 45.1694069
3: -15.9301043, 37.9201851, -12.5426521, 29.7492752, -45.6793747, 50.4628334
4: -16.8876266, 36.1860275, -13.2502241, 28.7574730, -45.6450996, 49.4362526

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B2_A1_B1_B2_A1

### Relational analysis result of NS_B1_A2_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5581364, upper bound: 43.5596421
time: 0.45 seconds

## Relational analysis of NS_B1_A2_B2_A1_B1_B2_A2

### Relational analysis result of NS_B1_A2_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5581364, upper bound: 43.5596421
time: 0.44 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -8.6322088, 32.3104172, -6.4340086, 24.4293156, -33.0615234, 38.7444153
1: -10.1549702, 36.7054749, -7.4736428, 27.6841717, -37.8391418, 44.1791153
2: -10.6975584, 36.7274017, -7.9794316, 27.7223930, -38.4199371, 44.7068329
3: -15.9301043, 37.9201851, -11.8481646, 28.4441738, -44.3742790, 49.7683487
4: -16.8876266, 36.1860275, -12.6650743, 27.2727757, -44.1603966, 48.8510971

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B2_A1_B2_B1_A1

### Relational analysis result of NS_B1_A2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5546630, upper bound: 43.5241353
time: 0.45 seconds

## Relational analysis of NS_B1_A2_B2_A1_B2_B1_A2

### Relational analysis result of NS_B1_A2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5546630, upper bound: 43.5241353
time: 0.45 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -8.6322088, 32.3104172, -6.9846678, 26.2288208, -34.8610268, 39.2950821
1: -10.1549702, 36.7054749, -8.1634932, 29.7061977, -39.8611603, 44.8689690
2: -10.6975584, 36.7274017, -8.6566067, 29.8250198, -40.5225792, 45.3840103
3: -15.9301043, 37.9201851, -12.8830853, 30.5647030, -46.4948044, 50.8032684
4: -16.8876266, 36.1860275, -13.6119404, 29.4874287, -46.3750534, 49.7979622

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B2_A1_B2_B2_A1

### Relational analysis result of NS_B1_A2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5546630, upper bound: 43.5241353
time: 0.44 seconds

## Relational analysis of NS_B1_A2_B2_A1_B2_B2_A2

### Relational analysis result of NS_B1_A2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5546630, upper bound: 43.5241353
time: 0.45 seconds

## BFS NS instance: NS_B1_A2_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -8.8174801, 33.0339127, -6.2620754, 23.7622166, -32.5796967, 39.2959900
1: -10.3915501, 37.5448494, -7.2577133, 26.9220257, -37.3135757, 44.8025589
2: -10.9315701, 37.5578690, -7.7621226, 26.9545021, -37.8860703, 45.3199921
3: -16.3067951, 38.8084564, -11.5024481, 27.6255684, -43.9323654, 50.3109016
4: -17.2786331, 36.9893837, -12.3027077, 26.5307827, -43.8094177, 49.2920914

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_A2_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A2_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A2_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B2_A2_B1_B1_A1

### Relational analysis result of NS_B1_A2_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5200645, upper bound: 43.5545765
time: 0.45 seconds

## Relational analysis of NS_B1_A2_B2_A2_B1_B1_A2

### Relational analysis result of NS_B1_A2_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5200645, upper bound: 43.5545765
time: 0.50 seconds

## BFS NS instance: NS_B1_A2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -8.8174801, 33.0339127, -6.8138876, 25.5695591, -34.3870392, 39.8477898
1: -10.3915501, 37.5448494, -7.9509349, 28.9397583, -39.3312950, 45.4957848
2: -10.9315701, 37.5578690, -8.4420090, 29.0706272, -40.0021973, 45.9998741
3: -16.3067951, 38.8084564, -12.5426521, 29.7492752, -46.0560608, 51.3511086
4: -17.2786331, 36.9893837, -13.2502241, 28.7574730, -46.0361061, 50.2396088

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B1_A2_B2_A2_B1_B2_B1

### Relational analysis result of NS_B1_A2_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4729601, upper bound: 43.5341837
time: 0.66 seconds

## Relational analysis of NS_B1_A2_B2_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B2_A2_B1_B2_A1

### Relational analysis result of NS_B1_A2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5200645, upper bound: 43.5545765
time: 0.53 seconds

## Relational analysis of NS_B1_A2_B2_A2_B1_B2_A2

### Relational analysis result of NS_B1_A2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5200645, upper bound: 43.5545765
time: 0.51 seconds

## BFS NS instance: NS_B1_A2_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -8.8174801, 33.0339127, -6.4340086, 24.4293156, -33.2467957, 39.4679108
1: -10.3915501, 37.5448494, -7.4736428, 27.6841717, -38.0757179, 45.0184898
2: -10.9315701, 37.5578690, -7.9794316, 27.7223930, -38.6539536, 45.5373001
3: -16.3067951, 38.8084564, -11.8481646, 28.4441738, -44.7509651, 50.6566200
4: -17.2786331, 36.9893837, -12.6650743, 27.2727757, -44.5514069, 49.6544571

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_A2_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B1_A2_B2_A2_B2_B1_B1

### Relational analysis result of NS_B1_A2_B2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4718604, upper bound: 43.5327660
time: 0.53 seconds

## Relational analysis of NS_B1_A2_B2_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A2_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B2_A2_B2_B1_A1

### Relational analysis result of NS_B1_A2_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5165910, upper bound: 43.5190697
time: 0.47 seconds

## Relational analysis of NS_B1_A2_B2_A2_B2_B1_A2

### Relational analysis result of NS_B1_A2_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5165910, upper bound: 43.5190697
time: 0.45 seconds

## BFS NS instance: NS_B1_A2_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -8.8174801, 33.0339127, -6.9846678, 26.2288208, -35.0463028, 40.0185776
1: -10.3915501, 37.5448494, -8.1634932, 29.7061977, -40.0977402, 45.7083435
2: -10.9315701, 37.5578690, -8.6566067, 29.8250198, -40.7565918, 46.2144775
3: -16.3067951, 38.8084564, -12.8830853, 30.5647030, -46.8714905, 51.6915436
4: -17.2786331, 36.9893837, -13.6119404, 29.4874287, -46.7660599, 50.6013260

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B1_A2_B2_A2_B2_B2_B1

### Relational analysis result of NS_B1_A2_B2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4694867, upper bound: 43.4986334
time: 0.48 seconds

## Relational analysis of NS_B1_A2_B2_A2_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A2_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B2_A2_B2_B2_A1

### Relational analysis result of NS_B1_A2_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5165910, upper bound: 43.5190697
time: 0.46 seconds

## Relational analysis of NS_B1_A2_B2_A2_B2_B2_A2

### Relational analysis result of NS_B1_A2_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5165910, upper bound: 43.5190697
time: 0.45 seconds

## BFS NS instance: NS_B2_A1_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -5.9647384, 22.4902878, -8.3524303, 31.3123322, -37.2770615, 30.8427162
1: -6.9048386, 25.4210529, -9.8154345, 35.5622025, -42.4670410, 35.2364845
2: -7.3848820, 25.5047226, -10.3523092, 35.5829659, -42.9678497, 35.8570213
3: -10.9337454, 26.0330429, -15.4090767, 36.7377586, -47.6715050, 41.4421120
4: -11.5946884, 25.2559605, -16.3657761, 35.0550346, -46.6497192, 41.6217270

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B2_A1_B1_A1_A1_B1_B1

### Relational analysis result of NS_B2_A1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5270486, upper bound: 43.5546874
time: 0.50 seconds

## Relational analysis of NS_B2_A1_B1_A1_A1_B1_B2

### Relational analysis result of NS_B2_A1_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5232082, upper bound: 43.5241833
time: 0.50 seconds

## BFS NS instance: NS_B2_A1_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -5.9647384, 22.4902878, -8.9357872, 33.2270203, -39.1917572, 31.4260750
1: -6.9048386, 25.4210529, -10.5515137, 37.7184982, -44.6233368, 35.9725647
2: -7.3848820, 25.5047226, -11.0725422, 37.8242493, -45.2091293, 36.5772438
3: -10.9337454, 26.0330429, -16.5153427, 38.9987679, -49.9325104, 42.5483780
4: -11.5946884, 25.2559605, -17.3748760, 37.4123497, -49.0070305, 42.6308365

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B2_A1_B1_A1_A1_B2_B1

### Relational analysis result of NS_B2_A1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5270486, upper bound: 43.5546874
time: 0.46 seconds

## Relational analysis of NS_B2_A1_B1_A1_A1_B2_B2

### Relational analysis result of NS_B2_A1_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5232082, upper bound: 43.5241833
time: 0.82 seconds

## BFS NS instance: NS_B2_A1_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -6.4972239, 24.2319641, -8.3524303, 31.3123322, -37.8095512, 32.5843849
1: -7.5682106, 27.3599796, -9.8154345, 35.5622025, -43.1304131, 37.1754112
2: -8.0439062, 27.5397072, -10.3523092, 35.5829659, -43.6268616, 37.8920174
3: -11.9258957, 28.0709877, -15.4090767, 36.7377586, -48.6636543, 43.4800644
4: -12.5062847, 27.4008636, -16.3657761, 35.0550346, -47.5613174, 43.7666321

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B2_A1_B1_A1_A2_B1_B1

### Relational analysis result of NS_B2_A1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5260811, upper bound: 43.5546812
time: 0.46 seconds

## Relational analysis of NS_B2_A1_B1_A1_A2_B1_B2

### Relational analysis result of NS_B2_A1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5222407, upper bound: 43.5241771
time: 0.50 seconds

## BFS NS instance: NS_B2_A1_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -6.4972239, 24.2319641, -8.9357872, 33.2270203, -39.7242432, 33.1677475
1: -7.5682106, 27.3599796, -10.5515137, 37.7184982, -45.2867088, 37.9114914
2: -8.0439062, 27.5397072, -11.0725422, 37.8242493, -45.8681488, 38.6122398
3: -11.9258957, 28.0709877, -16.5153427, 38.9987679, -50.9246597, 44.5863304
4: -12.5062847, 27.4008636, -17.3748760, 37.4123497, -49.9186325, 44.7757416

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B2_A1_B1_A1_A2_B2_B1

### Relational analysis result of NS_B2_A1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5260811, upper bound: 43.5546812
time: 0.46 seconds

## Relational analysis of NS_B2_A1_B1_A1_A2_B2_B2

### Relational analysis result of NS_B2_A1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5222407, upper bound: 43.5241771
time: 0.45 seconds

## BFS NS instance: NS_B2_A1_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -6.1316075, 23.1411781, -8.3524303, 31.3123322, -37.4439316, 31.4936047
1: -7.1131840, 26.1596413, -9.8154345, 35.5622025, -42.6753845, 35.9750748
2: -7.5950131, 26.2545033, -10.3523092, 35.5829659, -43.1779785, 36.6068115
3: -11.2687273, 26.8291836, -15.4090767, 36.7377586, -48.0064850, 42.2382469
4: -11.9468098, 25.9775772, -16.3657761, 35.0550346, -47.0018463, 42.3433456

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B2_A1_B1_A2_A1_B1_B1

### Relational analysis result of NS_B2_A1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5292472, upper bound: 43.5549981
time: 0.47 seconds

## Relational analysis of NS_B2_A1_B1_A2_A1_B1_B2

### Relational analysis result of NS_B2_A1_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5254068, upper bound: 43.5244940
time: 0.51 seconds

## BFS NS instance: NS_B2_A1_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -6.1316075, 23.1411781, -8.9357872, 33.2270203, -39.3586273, 32.0769653
1: -7.1131840, 26.1596413, -10.5515137, 37.7184982, -44.8316803, 36.7111511
2: -7.5950131, 26.2545033, -11.0725422, 37.8242493, -45.4192619, 37.3270340
3: -11.2687273, 26.8291836, -16.5153427, 38.9987679, -50.2674942, 43.3445129
4: -11.9468098, 25.9775772, -17.3748760, 37.4123497, -49.3591614, 43.3524513

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B2_A1_B1_A2_A1_B2_B1

### Relational analysis result of NS_B2_A1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5292472, upper bound: 43.5549981
time: 0.52 seconds

## Relational analysis of NS_B2_A1_B1_A2_A1_B2_B2

### Relational analysis result of NS_B2_A1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5254068, upper bound: 43.5244940
time: 0.50 seconds

## BFS NS instance: NS_B2_A1_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -6.6653934, 24.8864861, -8.3524303, 31.3123322, -37.9777184, 33.2389069
1: -7.7828097, 28.1194229, -9.8154345, 35.5622025, -43.3450050, 37.9348564
2: -8.2539454, 28.2906685, -10.3523092, 35.5829659, -43.8369102, 38.6429749
3: -12.2706099, 28.8829918, -15.4090767, 36.7377586, -49.0083694, 44.2920609
4: -12.8631258, 28.1254616, -16.3657761, 35.0550346, -47.9181595, 44.4912338

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B2_A1_B1_A2_A2_B1_B1

### Relational analysis result of NS_B2_A1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5241844, upper bound: 43.5546758
time: 0.46 seconds

## Relational analysis of NS_B2_A1_B1_A2_A2_B1_B2

### Relational analysis result of NS_B2_A1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5203440, upper bound: 43.5241717
time: 0.42 seconds

## BFS NS instance: NS_B2_A1_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -6.6653934, 24.8864861, -8.9357872, 33.2270203, -39.8924141, 33.8222694
1: -7.7828097, 28.1194229, -10.5515137, 37.7184982, -45.5013008, 38.6709366
2: -8.2539454, 28.2906685, -11.0725422, 37.8242493, -46.0781937, 39.3632011
3: -12.2706099, 28.8829918, -16.5153427, 38.9987679, -51.2693748, 45.3983307
4: -12.8631258, 28.1254616, -17.3748760, 37.4123497, -50.2754745, 45.5003357

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B2_A1_B1_A2_A2_B2_B1

### Relational analysis result of NS_B2_A1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5241844, upper bound: 43.5546758
time: 0.69 seconds

## Relational analysis of NS_B2_A1_B1_A2_A2_B2_B2

### Relational analysis result of NS_B2_A1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5203440, upper bound: 43.5241717
time: 0.47 seconds

## BFS NS instance: NS_B2_A2_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -8.4950399, 31.8271656, -8.5925446, 32.1775818, -40.6726227, 40.4197083
1: -9.9851398, 36.1547623, -10.1066513, 36.5556908, -46.5408325, 46.2614136
2: -10.5278149, 36.1718712, -10.6485958, 36.5736465, -47.1014595, 46.8204651
3: -15.6716042, 37.3449402, -15.8576345, 37.7638512, -53.4354553, 53.2025757
4: -16.6324043, 35.6286621, -16.8185234, 36.0260582, -52.6584625, 52.4471779

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_A1_B1_B1_A1_A1

### Relational analysis result of NS_B2_A2_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4196276, upper bound: 43.4490705
time: 0.52 seconds

## Relational analysis of NS_B2_A2_A1_B1_B1_A1_A2

### Relational analysis result of NS_B2_A2_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3642518, upper bound: 43.4426243
time: 0.60 seconds

## BFS NS instance: NS_B2_A2_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -8.5214157, 31.8919964, -8.5925446, 32.1775818, -40.6989937, 40.4845390
1: -10.0089788, 36.2651024, -10.1066513, 36.5556908, -46.5646667, 46.3717499
2: -10.5580931, 36.2347069, -10.6485958, 36.5736465, -47.1317329, 46.8833008
3: -15.7031174, 37.4892426, -15.8576345, 37.7638512, -53.4669685, 53.3468781
4: -16.7241974, 35.6652107, -16.8185234, 36.0260582, -52.7502556, 52.4837341

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_A1_B1_B1_A2_A1

### Relational analysis result of NS_B2_A2_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3823567, upper bound: 43.4446035
time: 0.59 seconds

## Relational analysis of NS_B2_A2_A1_B1_B1_A2_A2

### Relational analysis result of NS_B2_A2_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4167732, upper bound: 43.4448733
time: 0.68 seconds

## BFS NS instance: NS_B2_A2_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -8.3524303, 31.3123322, -8.7776985, 32.9010162, -41.2534447, 40.0900192
1: -9.8154345, 35.5622025, -10.3430977, 37.3951607, -47.2105904, 45.9053001
2: -10.3523092, 35.5829659, -10.8825235, 37.4039421, -47.7562523, 46.4654846
3: -15.4090767, 36.7377586, -16.2341270, 38.6523247, -54.0613976, 52.9718857
4: -16.3657761, 35.0550346, -17.2096958, 36.8289986, -53.1947670, 52.2647171

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B2_A2_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_A1_B1_B2_A1_B1

### Relational analysis result of NS_B2_A2_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4277048, upper bound: 43.4514339
time: 0.55 seconds

## Relational analysis of NS_B2_A2_A1_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_A1_B1_B2_A1_B1

### Relational analysis result of NS_B2_A2_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4256336, upper bound: 43.4514723
time: 0.52 seconds

## Relational analysis of NS_B2_A2_A1_B1_B2_A1_B2

### Relational analysis result of NS_B2_A2_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4256336, upper bound: 43.4514723
time: 0.52 seconds

## BFS NS instance: NS_B2_A2_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -8.9357872, 33.2270203, -8.7776985, 32.9010162, -41.8368034, 42.0047188
1: -10.5515137, 37.7184982, -10.3430977, 37.3951607, -47.9466705, 48.0615959
2: -11.0725422, 37.8242493, -10.8825235, 37.4039421, -48.4764748, 48.7067719
3: -16.5153427, 38.9987679, -16.2341270, 38.6523247, -55.1676636, 55.2328949
4: -17.3748760, 37.4123497, -17.2096958, 36.8289986, -54.2038727, 54.6220360

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B2_A2_A1_B1_B2_A2_A1

### Relational analysis result of NS_B2_A2_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3629291, upper bound: 43.4350564
time: 0.70 seconds

## Relational analysis of NS_B2_A2_A1_B1_B2_A2_A2

### Relational analysis result of NS_B2_A2_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4256336, upper bound: 43.4678438
time: 0.55 seconds

## BFS NS instance: NS_B2_A2_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -8.5925446, 32.1775818, -9.3598814, 35.1582031, -43.7507477, 41.5374641
1: -10.1066513, 36.5556908, -11.0692644, 40.0464897, -50.1222191, 47.6249542
2: -10.6485958, 36.5736465, -11.6192999, 39.9427032, -50.5912971, 48.1929474
3: -15.8576345, 37.7638512, -17.3481617, 41.4529648, -57.2642937, 55.1120110
4: -16.8185234, 36.0260582, -18.4878654, 39.2475166, -56.0660400, 54.5139236

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B2_A2_A1_B2_B2_A1_A1

### Relational analysis result of NS_B2_A2_A1_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3711829, upper bound: 43.3601952
time: 0.51 seconds

## Relational analysis of NS_B2_A2_A1_B2_B2_A1_A2

### Relational analysis result of NS_B2_A2_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3818490, upper bound: 43.3840848
time: 0.66 seconds

## BFS NS instance: NS_B2_A2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -9.1363468, 34.3769150, -9.3598814, 35.1582031, -44.2945480, 43.7367973
1: -10.7952032, 39.1587601, -11.0692644, 40.0464897, -50.8126144, 50.2046432
2: -11.3437166, 39.0394745, -11.6192999, 39.9427032, -51.2864189, 50.6587753
3: -16.9312782, 40.5276222, -17.3481617, 41.4529648, -58.3391075, 57.8351784
4: -18.0785217, 38.3399887, -18.4878654, 39.2475166, -57.3260384, 56.8278542

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_A1_B2_B2_A2_A1

### Relational analysis result of NS_B2_A2_A1_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -43.3797395, upper bound: 43.3572029
time: 0.95 seconds

## Relational analysis of NS_B2_A2_A1_B2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B2_A2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B2_A2_A1_B2_B2_A2_B1

### Relational analysis result of NS_B2_A2_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3601716, upper bound: 43.3833817
time: 0.87 seconds

## Relational analysis of NS_B2_A2_A1_B2_B2_A2_B2

### Relational analysis result of NS_B2_A2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3818490, upper bound: 43.3840848
time: 0.49 seconds

## BFS NS instance: NS_B2_A2_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -8.5439262, 32.0631142, -8.7506504, 32.7517700, -41.2956924, 40.8137589
1: -10.0602074, 36.4324837, -10.3020744, 37.2131882, -47.2733955, 46.7345581
2: -10.5933409, 36.4442482, -10.8460550, 37.2338791, -47.8272171, 47.2903023
3: -15.7987118, 37.6562881, -16.1563549, 38.4466438, -54.2453537, 53.8126373
4: -16.7670708, 35.8889656, -17.1193085, 36.6801529, -53.4472237, 53.0082741

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_A2_A1_B1_B1_B1

### Relational analysis result of NS_B2_A2_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4275308, upper bound: 43.4514301
time: 0.66 seconds

## Relational analysis of NS_B2_A2_A2_A1_B1_B1_B2

### Relational analysis result of NS_B2_A2_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4275308, upper bound: 43.4514301
time: 0.92 seconds

## BFS NS instance: NS_B2_A2_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -8.5439262, 32.0631142, -8.8175793, 33.0136490, -41.5575752, 40.8806915
1: -10.0602074, 36.4324837, -10.3814440, 37.5284348, -47.5886421, 46.8139191
2: -10.5933409, 36.4442482, -10.9269352, 37.5235748, -48.1169128, 47.3711853
3: -15.7987118, 37.6562881, -16.2779064, 38.7657852, -54.5644989, 53.9341888
4: -16.7670708, 35.8889656, -17.2583923, 36.9421082, -53.7091789, 53.1473579

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_A2_A1_B1_B2_B1

### Relational analysis result of NS_B2_A2_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4071380, upper bound: 43.4043641
time: 0.60 seconds

## Relational analysis of NS_B2_A2_A2_A1_B1_B2_B2

### Relational analysis result of NS_B2_A2_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.4071380, upper bound: 43.4043641
time: 0.52 seconds

## BFS NS instance: NS_B2_A2_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -9.0496149, 33.7122803, -8.8156204, 32.9617119, -42.0113258, 42.5279007
1: -10.6975231, 38.2898521, -10.3809900, 37.4477654, -48.1452866, 48.6708412
2: -11.2166643, 38.3750763, -10.9258604, 37.4781151, -48.6947784, 49.3009262
3: -16.7554111, 39.6034775, -16.2741280, 38.6946754, -55.4500847, 55.8776054
4: -17.6375637, 37.9246788, -17.2310925, 36.9354782, -54.5730438, 55.1557693

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_A2_A2_B1_A1_B1

### Relational analysis result of NS_B2_A2_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3667007, upper bound: 43.4186522
time: 0.85 seconds

## Relational analysis of NS_B2_A2_A2_A2_B1_A1_B2

### Relational analysis result of NS_B2_A2_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3667007, upper bound: 43.4186522
time: 0.69 seconds

## BFS NS instance: NS_B2_A2_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -9.1101408, 33.9550095, -8.8156204, 32.9617119, -42.0718536, 42.7706261
1: -10.7688112, 38.5767937, -10.3809900, 37.4477654, -48.2165756, 48.9577827
2: -11.2893696, 38.6422691, -10.9258604, 37.4781151, -48.7674751, 49.5681190
3: -16.8654156, 39.8994255, -16.2741280, 38.6946754, -55.5600891, 56.1735535
4: -17.7658501, 38.1622849, -17.2310925, 36.9354782, -54.7013283, 55.3933716

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_A2_A2_B1_A2_B1

### Relational analysis result of NS_B2_A2_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3991139, upper bound: 43.4184521
time: 0.61 seconds

## Relational analysis of NS_B2_A2_A2_A2_B1_A2_B2

### Relational analysis result of NS_B2_A2_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.3991139, upper bound: 43.4184521
time: 0.71 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.67 seconds
NS_B1_A1_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5735628, upper bound: 43.5624854
NS_B1_A1_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5274674, upper bound: 43.5577617
NS_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5321373, upper bound: 43.5914800
NS_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5274674, upper bound: 43.5577617
NS_B1_A1_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5711800, upper bound: 43.5298083
NS_B1_A1_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5250846, upper bound: 43.5250846
NS_B1_A1_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5711800, upper bound: 43.5298083
NS_B1_A1_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5250846, upper bound: 43.5250846
NS_B1_A1_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5624102, upper bound: 43.5735321
NS_B1_A1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5576866, upper bound: 43.5274367
NS_B1_A1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5624102, upper bound: 43.5735321
NS_B1_A1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5576866, upper bound: 43.5274367
NS_B1_A1_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5273146, upper bound: 43.5698741
NS_B1_A1_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5225909, upper bound: 43.5237787
NS_B1_A1_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5534529, upper bound: 43.5255783
NS_B1_A1_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5225909, upper bound: 43.5237787
NS_B1_A1_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5735321, upper bound: 43.5624102
NS_B1_A1_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5274367, upper bound: 43.5576866
NS_B1_A1_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5735321, upper bound: 43.5624102
NS_B1_A1_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5274367, upper bound: 43.5576866
NS_B1_A1_B2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5698741, upper bound: 43.5273146
NS_B1_A1_B2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5237787, upper bound: 43.5225909
NS_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5255783, upper bound: 43.5534529
NS_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5237787, upper bound: 43.5225909
NS_B1_A2_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5604773, upper bound: 43.5721456
NS_B1_A2_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5604773, upper bound: 43.5721456
NS_B1_A2_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5604773, upper bound: 43.5721456
NS_B1_A2_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5604773, upper bound: 43.5721456
NS_B1_A2_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5557536, upper bound: 43.5263719
NS_B1_A2_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5557536, upper bound: 43.5263719
NS_B1_A2_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5557536, upper bound: 43.5263719
NS_B1_A2_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5557536, upper bound: 43.5263719
NS_B1_A2_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5224054, upper bound: 43.5670800
NS_B1_A2_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5224054, upper bound: 43.5670800
NS_B1_A2_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5224054, upper bound: 43.5670800
NS_B1_A2_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5224054, upper bound: 43.5670800
NS_B1_A2_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5176817, upper bound: 43.5213063
NS_B1_A2_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5176817, upper bound: 43.5213063
NS_B1_A2_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5176817, upper bound: 43.5213063
NS_B1_A2_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5176817, upper bound: 43.5213063
NS_B1_A2_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5581364, upper bound: 43.5596421
NS_B1_A2_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5581364, upper bound: 43.5596421
NS_B1_A2_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5581364, upper bound: 43.5596421
NS_B1_A2_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5581364, upper bound: 43.5596421
NS_B1_A2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5546630, upper bound: 43.5241353
NS_B1_A2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5546630, upper bound: 43.5241353
NS_B1_A2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5546630, upper bound: 43.5241353
NS_B1_A2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5546630, upper bound: 43.5241353
NS_B1_A2_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5200645, upper bound: 43.5545765
NS_B1_A2_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5200645, upper bound: 43.5545765
NS_B1_A2_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5200645, upper bound: 43.5545765
NS_B1_A2_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5200645, upper bound: 43.5545765
NS_B1_A2_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5165910, upper bound: 43.5190697
NS_B1_A2_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5165910, upper bound: 43.5190697
NS_B1_A2_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5165910, upper bound: 43.5190697
NS_B1_A2_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5165910, upper bound: 43.5190697
NS_B2_A1_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5270486, upper bound: 43.5546874
NS_B2_A1_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5232082, upper bound: 43.5241833
NS_B2_A1_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5270486, upper bound: 43.5546874
NS_B2_A1_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5232082, upper bound: 43.5241833
NS_B2_A1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5260811, upper bound: 43.5546812
NS_B2_A1_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5222407, upper bound: 43.5241771
NS_B2_A1_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5260811, upper bound: 43.5546812
NS_B2_A1_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5222407, upper bound: 43.5241771
NS_B2_A1_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5292472, upper bound: 43.5549981
NS_B2_A1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5254068, upper bound: 43.5244940
NS_B2_A1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5292472, upper bound: 43.5549981
NS_B2_A1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5254068, upper bound: 43.5244940
NS_B2_A1_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5241844, upper bound: 43.5546758
NS_B2_A1_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5203440, upper bound: 43.5241717
NS_B2_A1_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5241844, upper bound: 43.5546758
NS_B2_A1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.5203440, upper bound: 43.5241717
NS_B2_A2_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.4196276, upper bound: 43.4490705
NS_B2_A2_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.3642518, upper bound: 43.4426243
NS_B2_A2_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.3823567, upper bound: 43.4446035
NS_B2_A2_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.4167732, upper bound: 43.4448733
NS_B2_A2_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.4256336, upper bound: 43.4514723
NS_B2_A2_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.4256336, upper bound: 43.4514723
NS_B2_A2_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.3629291, upper bound: 43.4350564
NS_B2_A2_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.4256336, upper bound: 43.4678438
NS_B2_A2_A1_B2_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.3711829, upper bound: 43.3601952
NS_B2_A2_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.3818490, upper bound: 43.3840848
NS_B2_A2_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.3601716, upper bound: 43.3833817
NS_B2_A2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.3818490, upper bound: 43.3840848
NS_B2_A2_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.4275308, upper bound: 43.4514301
NS_B2_A2_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.4275308, upper bound: 43.4514301
NS_B2_A2_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.4071380, upper bound: 43.4043641
NS_B2_A2_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.4071380, upper bound: 43.4043641
NS_B2_A2_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.3667007, upper bound: 43.4186522
NS_B2_A2_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.3667007, upper bound: 43.4186522
NS_B2_A2_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.3991139, upper bound: 43.4184521
NS_B2_A2_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.67
Output dim: 3, lower bound: -43.3991139, upper bound: 43.4184521

## BFS NS instance: NS_B1_A1_B1_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -5.5454221, 21.0587044, -5.8062859, 21.9969387, -27.5423603, 26.8649902
1: -6.4011383, 23.8442497, -6.7116928, 24.9162731, -31.3174095, 30.5559387
2: -6.8633947, 23.8585110, -7.1873412, 24.9264069, -31.7898006, 31.0458527
3: -10.1740608, 24.3810234, -10.6498022, 25.4913673, -35.6654282, 35.0308266
4: -10.8713808, 23.5251331, -11.3617077, 24.5736980, -35.4450798, 34.8868408

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B1_A1_B1_A1_A1_B1

### Relational analysis result of NS_B1_A1_B1_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5601445, upper bound: 43.5601445
time: 0.46 seconds

## Relational analysis of NS_B1_A1_B1_A1_B1_A1_A1_B2

### Relational analysis result of NS_B1_A1_B1_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5601445, upper bound: 43.5601445
time: 0.51 seconds

## BFS NS instance: NS_B1_A1_B1_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -6.0728350, 22.7863827, -5.8062859, 21.9969387, -28.0697708, 28.5926666
1: -7.0567961, 25.7621460, -6.7116928, 24.9162731, -31.9730682, 32.4738388
2: -7.5153069, 25.8768997, -7.1873412, 24.9264069, -32.4417152, 33.0642395
3: -11.1538324, 26.3902988, -10.6498022, 25.4913673, -36.6451912, 37.0401001
4: -11.7746954, 25.6467304, -11.3617077, 24.5736980, -36.3483925, 37.0084381

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_B1_A1_B1_A1_A2_A1

### Relational analysis result of NS_B1_A1_B1_A1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5555346, upper bound: 43.5230528
time: 0.62 seconds

## Relational analysis of NS_B1_A1_B1_A1_B1_A1_A2_A2

### Relational analysis result of NS_B1_A1_B1_A1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5190999, upper bound: 43.5190999
time: 0.69 seconds

## BFS NS instance: NS_B1_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -5.9676609, 22.6236305, -5.5454221, 21.0587044, -27.0263615, 28.1690521
1: -6.9145479, 25.6272087, -6.4011383, 23.8442497, -30.7587967, 32.0283432
2: -7.3910317, 25.6533833, -6.8633947, 23.8585110, -31.2495422, 32.5167770
3: -10.9741497, 26.2476044, -10.1740608, 24.3810234, -35.3551712, 36.4216576
4: -11.7002325, 25.2735844, -10.8713808, 23.5251331, -35.2253647, 36.1449661

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_B1_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5274674, upper bound: 43.5577617
time: 0.44 seconds

## Relational analysis of NS_B1_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_B1_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5274674, upper bound: 43.5577617
time: 0.49 seconds

## BFS NS instance: NS_B1_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -5.9676609, 22.6236305, -6.0728350, 22.7863827, -28.7540398, 28.6964626
1: -6.9145479, 25.6272087, -7.0567961, 25.7621460, -32.6766930, 32.6840057
2: -7.3910317, 25.6533833, -7.5153069, 25.8768997, -33.2679329, 33.1686897
3: -10.9741497, 26.2476044, -11.1538324, 26.3902988, -37.3644485, 37.4014244
4: -11.7002325, 25.2735844, -11.7746954, 25.6467304, -37.3469620, 37.0482788

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_B1_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5274674, upper bound: 43.5577617
time: 0.47 seconds

## Relational analysis of NS_B1_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_B1_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5274674, upper bound: 43.5577617
time: 0.50 seconds

## BFS NS instance: NS_B1_A1_B1_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -5.5454221, 21.0587044, -5.9676609, 22.6236305, -28.1690521, 27.0263615
1: -6.4011383, 23.8442497, -6.9145479, 25.6272087, -32.0283432, 30.7587967
2: -6.8633947, 23.8585110, -7.3910317, 25.6533833, -32.5167770, 31.2495422
3: -10.1740608, 24.3810234, -10.9741497, 26.2476044, -36.4216576, 35.3551712
4: -10.8713808, 23.5251331, -11.7002325, 25.2735844, -36.1449661, 35.2253647

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B1_A1_B2_A1_A1_B1

### Relational analysis result of NS_B1_A1_B1_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5577617, upper bound: 43.5274674
time: 0.60 seconds

## Relational analysis of NS_B1_A1_B1_A1_B2_A1_A1_B2

### Relational analysis result of NS_B1_A1_B1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5577617, upper bound: 43.5274674
time: 0.48 seconds

## BFS NS instance: NS_B1_A1_B1_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -6.0728350, 22.7863827, -5.9676609, 22.6236305, -28.6964626, 28.7540398
1: -7.0567961, 25.7621460, -6.9145479, 25.6272087, -32.6840057, 32.6766930
2: -7.5153069, 25.8768997, -7.3910317, 25.6533833, -33.1686897, 33.2679329
3: -11.1538324, 26.3902988, -10.9741497, 26.2476044, -37.4014244, 37.3644485
4: -11.7746954, 25.6467304, -11.7002325, 25.2735844, -37.0482788, 37.3469620

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_B1_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B1_A1_B2_A1_A2_B1

### Relational analysis result of NS_B1_A1_B1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5577617, upper bound: 43.5274674
time: 0.49 seconds

## Relational analysis of NS_B1_A1_B1_A1_B2_A1_A2_B2

### Relational analysis result of NS_B1_A1_B1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5577617, upper bound: 43.5274674
time: 0.46 seconds

## BFS NS instance: NS_B1_A1_B1_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -5.7107072, 21.7049503, -5.9676609, 22.6236305, -28.3343372, 27.6726093
1: -6.6079497, 24.5724201, -6.9145479, 25.6272087, -32.2351570, 31.4869690
2: -7.0722151, 24.6020393, -7.3910317, 25.6533833, -32.7255974, 31.9930706
3: -10.5045061, 25.1580029, -10.9741497, 26.2476044, -36.7521057, 36.1321526
4: -11.2190008, 24.2401142, -11.7002325, 25.2735844, -36.4925842, 35.9403419

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B1_A1_B2_A2_A1_B1

### Relational analysis result of NS_B1_A1_B1_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5250846, upper bound: 43.5250846
time: 0.46 seconds

## Relational analysis of NS_B1_A1_B1_A1_B2_A2_A1_B2

### Relational analysis result of NS_B1_A1_B1_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5250846, upper bound: 43.5250846
time: 0.45 seconds

## BFS NS instance: NS_B1_A1_B1_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -6.2229776, 23.3687820, -5.9676609, 22.6236305, -28.8466072, 29.3364410
1: -7.2409058, 26.4228458, -6.9145479, 25.6272087, -32.8681107, 33.3373909
2: -7.7032561, 26.5428905, -7.3910317, 25.6533833, -33.3566399, 33.9339180
3: -11.4496880, 27.1048088, -10.9741497, 26.2476044, -37.6972885, 38.0789566
4: -12.0935068, 26.2858315, -11.7002325, 25.2735844, -37.3670883, 37.9860535

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B1_A1_B2_A2_A2_B1

### Relational analysis result of NS_B1_A1_B1_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5250846, upper bound: 43.5250846
time: 0.62 seconds

## Relational analysis of NS_B1_A1_B1_A1_B2_A2_A2_B2

### Relational analysis result of NS_B1_A1_B1_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5250846, upper bound: 43.5250846
time: 0.75 seconds

## BFS NS instance: NS_B1_A1_B1_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -6.3634715, 24.0477009, -5.5454221, 21.0587044, -27.4221745, 29.5931225
1: -7.3651905, 27.2656727, -6.4011383, 23.8442497, -31.2094402, 33.6668091
2: -7.8789954, 27.2677956, -6.8633947, 23.8585110, -31.7375069, 34.1311913
3: -11.6574259, 27.9324017, -10.1740608, 24.3810234, -36.0384483, 38.1064529
4: -12.4452915, 26.8532047, -10.8713808, 23.5251331, -35.9704247, 37.7245865

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B1_A2_A1_B1_B1_A1

### Relational analysis result of NS_B1_A1_B1_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5600693, upper bound: 43.5601138
time: 0.49 seconds

## Relational analysis of NS_B1_A1_B1_A2_A1_B1_B1_A2

### Relational analysis result of NS_B1_A1_B1_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5600693, upper bound: 43.5601138
time: 0.50 seconds

## BFS NS instance: NS_B1_A1_B1_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -6.3634715, 24.0477009, -6.0728350, 22.7863827, -29.1498528, 30.1205330
1: -7.3651905, 27.2656727, -7.0567961, 25.7621460, -33.1273346, 34.3224678
2: -7.8789954, 27.2677956, -7.5153069, 25.8768997, -33.7558937, 34.7831039
3: -11.6574259, 27.9324017, -11.1538324, 26.3902988, -38.0477257, 39.0862236
4: -12.4452915, 26.8532047, -11.7746954, 25.6467304, -38.0920219, 38.6278992

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B1_A2_A1_B1_B2_A1

### Relational analysis result of NS_B1_A1_B1_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5600693, upper bound: 43.5601138
time: 0.45 seconds

## Relational analysis of NS_B1_A1_B1_A2_A1_B1_B2_A2

### Relational analysis result of NS_B1_A1_B1_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5600693, upper bound: 43.5601138
time: 0.49 seconds

## BFS NS instance: NS_B1_A1_B1_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -6.3634715, 24.0477009, -5.7107072, 21.7049503, -28.0684223, 29.7584076
1: -7.3651905, 27.2656727, -6.6079497, 24.5724201, -31.9376106, 33.8736229
2: -7.8789954, 27.2677956, -7.0722151, 24.6020393, -32.4810333, 34.3400116
3: -11.6574259, 27.9324017, -10.5045061, 25.1580029, -36.8154297, 38.4369049
4: -12.4452915, 26.8532047, -11.2190008, 24.2401142, -36.6854019, 38.0722046

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B1_A2_A1_B2_B1_A1

### Relational analysis result of NS_B1_A1_B1_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5576866, upper bound: 43.5274367
time: 0.46 seconds

## Relational analysis of NS_B1_A1_B1_A2_A1_B2_B1_A2

### Relational analysis result of NS_B1_A1_B1_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5576866, upper bound: 43.5274367
time: 0.49 seconds

## BFS NS instance: NS_B1_A1_B1_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -6.3634715, 24.0477009, -6.2229776, 23.3687820, -29.7322540, 30.2706795
1: -7.3651905, 27.2656727, -7.2409058, 26.4228458, -33.7880363, 34.5065765
2: -7.8789954, 27.2677956, -7.7032561, 26.5428905, -34.4218826, 34.9710503
3: -11.6574259, 27.9324017, -11.4496880, 27.1048088, -38.7622337, 39.3820839
4: -12.4452915, 26.8532047, -12.0935068, 26.2858315, -38.7311172, 38.9467125

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B1_A2_A1_B2_B2_A1

### Relational analysis result of NS_B1_A1_B1_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5576866, upper bound: 43.5274367
time: 0.64 seconds

## Relational analysis of NS_B1_A1_B1_A2_A1_B2_B2_A2

### Relational analysis result of NS_B1_A1_B1_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5576866, upper bound: 43.5274367
time: 0.62 seconds

## BFS NS instance: NS_B1_A1_B1_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -6.5370998, 24.7172279, -5.5454221, 21.0587044, -27.5958023, 30.2626495
1: -7.5843964, 28.0217705, -6.4011383, 23.8442497, -31.4286461, 34.4229088
2: -8.0978727, 28.0383968, -6.8633947, 23.8585110, -31.9563828, 34.9017906
3: -12.0083332, 28.7461624, -10.1740608, 24.3810234, -36.3893585, 38.9202194
4: -12.8104057, 27.6020432, -10.8713808, 23.5251331, -36.3355331, 38.4734230

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A1_B1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B1_A2_A2_B1_B1_A1

### Relational analysis result of NS_B1_A1_B1_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5249737, upper bound: 43.5564559
time: 0.46 seconds

## Relational analysis of NS_B1_A1_B1_A2_A2_B1_B1_A2

### Relational analysis result of NS_B1_A1_B1_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5249737, upper bound: 43.5564559
time: 0.49 seconds

## BFS NS instance: NS_B1_A1_B1_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -6.5370998, 24.7172279, -6.0728350, 22.7863827, -29.3234806, 30.7900620
1: -7.5843964, 28.0217705, -7.0567961, 25.7621460, -33.3465424, 35.0785675
2: -8.0978727, 28.0383968, -7.5153069, 25.8768997, -33.9747734, 35.5537033
3: -12.0083332, 28.7461624, -11.1538324, 26.3902988, -38.3986320, 39.8999863
4: -12.8104057, 27.6020432, -11.7746954, 25.6467304, -38.4571381, 39.3767395

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_B1_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B1_A2_A2_B1_B2_A1

### Relational analysis result of NS_B1_A1_B1_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5249737, upper bound: 43.5564559
time: 0.68 seconds

## Relational analysis of NS_B1_A1_B1_A2_A2_B1_B2_A2

### Relational analysis result of NS_B1_A1_B1_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5249737, upper bound: 43.5564559
time: 0.72 seconds

## BFS NS instance: NS_B1_A1_B1_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -6.2745233, 23.7754993, -5.9676609, 22.6236305, -28.8981533, 29.7431602
1: -7.2676334, 26.9373245, -6.9145479, 25.6272087, -32.8948441, 33.8518639
2: -7.7722998, 26.9597549, -7.3910317, 25.6533833, -33.4256783, 34.3507843
3: -11.5222073, 27.6287766, -10.9741497, 26.2476044, -37.7697983, 38.6029243
4: -12.3165798, 26.5360985, -11.7002325, 25.2735844, -37.5901527, 38.2363281

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B1_A2_A2_B2_A1_B1

### Relational analysis result of NS_B1_A1_B1_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5225909, upper bound: 43.5237787
time: 0.50 seconds

## Relational analysis of NS_B1_A1_B1_A2_A2_B2_A1_B2

### Relational analysis result of NS_B1_A1_B1_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5225909, upper bound: 43.5237787
time: 0.84 seconds

## BFS NS instance: NS_B1_A1_B1_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6.8090243, 25.5240231, -5.9676609, 22.6236305, -29.4326534, 31.4916840
1: -7.9357371, 28.8916550, -6.9145479, 25.6272087, -33.5629425, 35.8062019
2: -8.4312859, 29.0014076, -7.3910317, 25.6533833, -34.0846710, 36.3924408
3: -12.5244751, 29.6865978, -10.9741497, 26.2476044, -38.7720718, 40.6607475
4: -13.2364388, 28.6882744, -11.7002325, 25.2735844, -38.5100174, 40.3885078

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B1_A2_A2_B2_A2_B1

### Relational analysis result of NS_B1_A1_B1_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5225909, upper bound: 43.5237787
time: 0.52 seconds

## Relational analysis of NS_B1_A1_B1_A2_A2_B2_A2_B2

### Relational analysis result of NS_B1_A1_B1_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5225909, upper bound: 43.5237787
time: 0.49 seconds

## BFS NS instance: NS_B1_A1_B2_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -5.5454221, 21.0587044, -6.3634715, 24.0477009, -29.5931225, 27.4221745
1: -6.4011383, 23.8442497, -7.3651905, 27.2656727, -33.6668091, 31.2094402
2: -6.8633947, 23.8585110, -7.8789954, 27.2677956, -34.1311913, 31.7375069
3: -10.1740608, 24.3810234, -11.6574259, 27.9324017, -38.1064529, 36.0384483
4: -10.8713808, 23.5251331, -12.4452915, 26.8532047, -37.7245865, 35.9704247

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B2_A1_B1_A1_A1_B1

### Relational analysis result of NS_B1_A1_B2_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5601138, upper bound: 43.5600693
time: 0.47 seconds

## Relational analysis of NS_B1_A1_B2_A1_B1_A1_A1_B2

### Relational analysis result of NS_B1_A1_B2_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5601138, upper bound: 43.5600693
time: 0.46 seconds

## BFS NS instance: NS_B1_A1_B2_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -6.0728350, 22.7863827, -6.3634715, 24.0477009, -30.1205330, 29.1498528
1: -7.0567961, 25.7621460, -7.3651905, 27.2656727, -34.3224678, 33.1273346
2: -7.5153069, 25.8768997, -7.8789954, 27.2677956, -34.7831039, 33.7558937
3: -11.1538324, 26.3902988, -11.6574259, 27.9324017, -39.0862236, 38.0477257
4: -11.7746954, 25.6467304, -12.4452915, 26.8532047, -38.6278992, 38.0920219

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_B2_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B2_A1_B1_A1_A2_B1

### Relational analysis result of NS_B1_A1_B2_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5601138, upper bound: 43.5600693
time: 0.43 seconds

## Relational analysis of NS_B1_A1_B2_A1_B1_A1_A2_B2

### Relational analysis result of NS_B1_A1_B2_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5601138, upper bound: 43.5600693
time: 0.44 seconds

## BFS NS instance: NS_B1_A1_B2_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -5.7107072, 21.7049503, -6.3634715, 24.0477009, -29.7584076, 28.0684223
1: -6.6079497, 24.5724201, -7.3651905, 27.2656727, -33.8736229, 31.9376106
2: -7.0722151, 24.6020393, -7.8789954, 27.2677956, -34.3400116, 32.4810333
3: -10.5045061, 25.1580029, -11.6574259, 27.9324017, -38.4369049, 36.8154297
4: -11.2190008, 24.2401142, -12.4452915, 26.8532047, -38.0722046, 36.6854019

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B2_A1_B1_A2_A1_B1

### Relational analysis result of NS_B1_A1_B2_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5274367, upper bound: 43.5576866
time: 0.49 seconds

## Relational analysis of NS_B1_A1_B2_A1_B1_A2_A1_B2

### Relational analysis result of NS_B1_A1_B2_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5274367, upper bound: 43.5576866
time: 0.48 seconds

## BFS NS instance: NS_B1_A1_B2_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -6.2229776, 23.3687820, -6.3634715, 24.0477009, -30.2706795, 29.7322540
1: -7.2409058, 26.4228458, -7.3651905, 27.2656727, -34.5065765, 33.7880363
2: -7.7032561, 26.5428905, -7.8789954, 27.2677956, -34.9710503, 34.4218788
3: -11.4496880, 27.1048088, -11.6574259, 27.9324017, -39.3820839, 38.7622337
4: -12.0935068, 26.2858315, -12.4452915, 26.8532047, -38.9467125, 38.7311172

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B2_A1_B1_A2_A2_B1

### Relational analysis result of NS_B1_A1_B2_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5274367, upper bound: 43.5576866
time: 0.50 seconds

## Relational analysis of NS_B1_A1_B2_A1_B1_A2_A2_B2

### Relational analysis result of NS_B1_A1_B2_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5274367, upper bound: 43.5576866
time: 0.49 seconds

## BFS NS instance: NS_B1_A1_B2_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -5.5454221, 21.0587044, -6.5370998, 24.7172279, -30.2626495, 27.5958023
1: -6.4011383, 23.8442497, -7.5843964, 28.0217705, -34.4229088, 31.4286442
2: -6.8633947, 23.8585110, -8.0978727, 28.0383968, -34.9017906, 31.9563828
3: -10.1740608, 24.3810234, -12.0083332, 28.7461624, -38.9202194, 36.3893585
4: -10.8713808, 23.5251331, -12.8104057, 27.6020432, -38.4734230, 36.3355331

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B1_A1_B2_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B2_A1_B2_A1_A1_B1

### Relational analysis result of NS_B1_A1_B2_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5564559, upper bound: 43.5249737
time: 0.70 seconds

## Relational analysis of NS_B1_A1_B2_A1_B2_A1_A1_B2

### Relational analysis result of NS_B1_A1_B2_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5564559, upper bound: 43.5249737
time: 0.48 seconds

## BFS NS instance: NS_B1_A1_B2_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -6.0728350, 22.7863827, -6.5370998, 24.7172279, -30.7900620, 29.3234806
1: -7.0567961, 25.7621460, -7.5843964, 28.0217705, -35.0785675, 33.3465424
2: -7.5153069, 25.8768997, -8.0978727, 28.0383968, -35.5537033, 33.9747734
3: -11.1538324, 26.3902988, -12.0083332, 28.7461624, -39.8999863, 38.3986320
4: -11.7746954, 25.6467304, -12.8104057, 27.6020432, -39.3767395, 38.4571381

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_B2_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B2_A1_B2_A1_A2_B1

### Relational analysis result of NS_B1_A1_B2_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5564559, upper bound: 43.5249737
time: 0.46 seconds

## Relational analysis of NS_B1_A1_B2_A1_B2_A1_A2_B2

### Relational analysis result of NS_B1_A1_B2_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5564559, upper bound: 43.5249737
time: 0.47 seconds

## BFS NS instance: NS_B1_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -5.9676609, 22.6236305, -6.2745233, 23.7754993, -29.7431602, 28.8981533
1: -6.9145479, 25.6272087, -7.2676334, 26.9373245, -33.8518639, 32.8948441
2: -7.3910317, 25.6533833, -7.7722998, 26.9597549, -34.3507843, 33.4256821
3: -10.9741497, 26.2476044, -11.5222073, 27.6287766, -38.6029282, 37.7697983
4: -11.7002325, 25.2735844, -12.3165798, 26.5360985, -38.2363281, 37.5901527

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_B1_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5237787, upper bound: 43.5225909
time: 0.50 seconds

## Relational analysis of NS_B1_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_B1_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5237787, upper bound: 43.5225909
time: 0.48 seconds

## BFS NS instance: NS_B1_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -5.9676609, 22.6236305, -6.8090243, 25.5240231, -31.4916840, 29.4326553
1: -6.9145479, 25.6272087, -7.9357371, 28.8916550, -35.8061981, 33.5629387
2: -7.3910317, 25.6533833, -8.4312859, 29.0014076, -36.3924408, 34.0846710
3: -10.9741497, 26.2476044, -12.5244751, 29.6865978, -40.6607475, 38.7720718
4: -11.7002325, 25.2735844, -13.2364388, 28.6882744, -40.3885078, 38.5100174

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_B1_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5237787, upper bound: 43.5225909
time: 0.46 seconds

## Relational analysis of NS_B1_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_B1_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5237787, upper bound: 43.5225909
time: 0.45 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -8.3524303, 31.3123322, -4.8395281, 18.6394978, -26.9919243, 36.1518555
1: -9.8154345, 35.5622025, -5.5451407, 21.1490383, -30.9644699, 41.1073418
2: -10.3523092, 35.5829659, -5.9836831, 21.0349770, -31.3872871, 41.5666504
3: -15.4090767, 36.7377586, -8.8850317, 21.5857143, -36.9947853, 45.6227913
4: -16.3657761, 35.0550346, -9.6547794, 20.5678463, -36.9336128, 44.7098122

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_A1_A1

### Relational analysis result of NS_B1_A2_B1_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5546874, upper bound: 43.5243307
time: 0.49 seconds

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_A1_A2

### Relational analysis result of NS_B1_A2_B1_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -43.5241833, upper bound: 43.5204903
time: 0.66 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -8.9357872, 33.2270203, -4.8395281, 18.6394978, -27.5752850, 38.0665474
1: -10.5515137, 37.7184982, -5.5451407, 21.1490383, -31.7005520, 43.2636375
2: -11.0725422, 37.8242493, -5.9836831, 21.0349770, -32.1075172, 43.8079338
3: -16.5153427, 38.9987679, -8.8850317, 21.5857143, -38.1010513, 47.8838005
4: -17.3748760, 37.4123497, -9.6547794, 20.5678463, -37.9427223, 47.0671272

Time for backsubstitution: 1.84 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.05 + 418.69 = 421.75 seconds
