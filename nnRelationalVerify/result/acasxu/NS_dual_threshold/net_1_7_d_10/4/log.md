## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_7.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 38.7593690755


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.4935045, 16.7457676, -4.4935045, 16.7457676, -21.2392731, 21.2392712)
1: (-12.6764650, 34.5451698, -12.6764650, 34.5451698, -47.2216339, 47.2216339)
2: (-19.5602589, 31.8633156, -19.5602589, 31.8633156, -51.4235764, 51.4235764)
3: (-10.6092863, 40.3084297, -10.6092863, 40.3084297, -50.9177132, 50.9177170)
4: (-17.5911102, 27.9587860, -17.5911102, 27.9587860, -45.5498962, 45.5498962)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.34 + 2.28 = 3.62 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -38.7671225, upper bound: 38.7671225

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7669686, upper bound: 38.7652505
time: 1.20 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7671225, upper bound: 38.7671225
time: 0.68 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.01 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 2.01
Output dim: 4, lower bound: -38.7669686, upper bound: 38.7652505
NS_B2, status: Status.UNKNOWN, split count: 1, time: 2.01
Output dim: 4, lower bound: -38.7671225, upper bound: 38.7671225

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -4.4665918, 16.6453495, -3.7629883, 14.0485401, -18.5151310, 20.4083385
1: -12.6012287, 34.3388596, -10.6152039, 28.9633255, -41.5645485, 44.9540634
2: -19.4450188, 31.6718750, -16.3718643, 26.6367149, -46.0817337, 48.0437241
3: -10.5457926, 40.0718880, -8.8767414, 33.8340149, -44.3798065, 48.9486313
4: -17.4865170, 27.7910957, -14.7028122, 23.3590126, -40.8455276, 42.4939079

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7650966, upper bound: 38.7650966
time: 0.65 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7650966, upper bound: 38.7652505
time: 0.66 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -4.4935045, 16.7457676, -4.4397011, 16.5434704, -21.0369759, 21.1854668
1: -12.6764650, 34.5451698, -12.5246544, 34.1262398, -46.8027000, 47.0698204
2: -19.5602589, 31.8633156, -19.3276100, 31.4766617, -51.0369186, 51.1909256
3: -10.6092863, 40.3084297, -10.4815588, 39.8213959, -50.4306717, 50.7899895
4: -17.5911102, 27.9587860, -17.3803291, 27.6185970, -45.2097092, 45.3391151

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7652505, upper bound: 38.7669686
time: 0.60 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7652505, upper bound: 38.7671225
time: 0.60 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.16 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 4, lower bound: -38.7650966, upper bound: 38.7650966
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 4, lower bound: -38.7650966, upper bound: 38.7652505
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 4, lower bound: -38.7652505, upper bound: 38.7669686
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 4, lower bound: -38.7652505, upper bound: 38.7671225

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -3.7629883, 14.0485401, -3.7629883, 14.0485401, -17.8115292, 17.8115292
1: -10.6152039, 28.9633255, -10.6152039, 28.9633255, -39.5785217, 39.5785255
2: -16.3718643, 26.6367149, -16.3718643, 26.6367149, -43.0085754, 43.0085793
3: -8.8767414, 33.8340149, -8.8767414, 33.8340149, -42.7107544, 42.7107544
4: -14.7028122, 23.3590126, -14.7028122, 23.3590126, -38.0618172, 38.0618210

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A1_A1

### Relational analysis result of NS_B1_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7226231, upper bound: 38.7489006
time: 0.63 seconds

## Relational analysis of NS_B1_A1_A2

### Relational analysis result of NS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7650966, upper bound: 38.7650966
time: 0.70 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -4.4397011, 16.5434704, -3.7629883, 14.0485401, -18.4882412, 20.3064594
1: -12.5246544, 34.1262398, -10.6152039, 28.9633255, -41.4879684, 44.7414436
2: -19.3276100, 31.4766617, -16.3718643, 26.6367149, -45.9643250, 47.8485260
3: -10.4815588, 39.8213959, -8.8767414, 33.8340149, -44.3155746, 48.6981354
4: -17.3803291, 27.6185970, -14.7028122, 23.3590126, -40.7393379, 42.3214073

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7632011, upper bound: 38.7516429
time: 1.17 seconds

## Relational analysis of NS_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7632011, upper bound: 38.7652505
time: 1.12 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -3.7629883, 14.0485401, -4.4397011, 16.5434704, -20.3064594, 18.4882412
1: -10.6152039, 28.9633255, -12.5246544, 34.1262398, -44.7414436, 41.4879684
2: -16.3718643, 26.6367149, -19.3276100, 31.4766617, -47.8485260, 45.9643250
3: -8.8767414, 33.8340149, -10.4815588, 39.8213959, -48.6981354, 44.3155746
4: -14.7028122, 23.3590126, -17.3803291, 27.6185970, -42.3214111, 40.7393417

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 18

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_B1

### Relational analysis result of NS_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7485574, upper bound: 38.7503040
time: 0.87 seconds

## Relational analysis of NS_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7650966, upper bound: 38.7668775
time: 1.38 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -4.4397011, 16.5434704, -4.4397011, 16.5434704, -20.9831715, 20.9831715
1: -12.5246544, 34.1262398, -12.5246544, 34.1262398, -46.6508865, 46.6508865
2: -19.3276100, 31.4766617, -19.3276100, 31.4766617, -50.8042717, 50.8042717
3: -10.4815588, 39.8213959, -10.4815588, 39.8213959, -50.3029480, 50.3029518
4: -17.3803291, 27.6185970, -17.3803291, 27.6185970, -44.9989243, 44.9989243

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_A1

### Relational analysis result of NS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7224665, upper bound: 38.7636174
time: 0.84 seconds

## Relational analysis of NS_B2_A2_A2

### Relational analysis result of NS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7650966, upper bound: 38.7670314
time: 1.60 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.64 seconds
NS_B1_A1_A1, status: Status.VERIFIED, split count: 3, time: 3.64
Output dim: 4, lower bound: -38.7226231, upper bound: 38.7489006
NS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 3.64
Output dim: 4, lower bound: -38.7650966, upper bound: 38.7650966
NS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.64
Output dim: 4, lower bound: -38.7632011, upper bound: 38.7516429
NS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.64
Output dim: 4, lower bound: -38.7632011, upper bound: 38.7652505
NS_B2_A1_B1, status: Status.VERIFIED, split count: 3, time: 3.64
Output dim: 4, lower bound: -38.7485574, upper bound: 38.7503040
NS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.64
Output dim: 4, lower bound: -38.7650966, upper bound: 38.7668775
NS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 3.64
Output dim: 4, lower bound: -38.7224665, upper bound: 38.7636174
NS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.64
Output dim: 4, lower bound: -38.7650966, upper bound: 38.7670314

## BFS NS instance: NS_B1_A1_A2

### Backsubstitution after applying NS history:
0: -3.7267268, 13.9167728, -3.7588351, 14.0336018, -17.7603283, 17.6756077
1: -10.5112848, 28.6832466, -10.6033792, 28.9314671, -39.4427528, 39.2866173
2: -16.2075958, 26.3915825, -16.3532505, 26.6089439, -42.8165398, 42.7448349
3: -8.7892141, 33.5076637, -8.8667536, 33.7970123, -42.5862198, 42.3744125
4: -14.5561237, 23.1437645, -14.6861916, 23.3346539, -37.8907776, 37.8299561

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A1_A2_B1

### Relational analysis result of NS_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7489006, upper bound: 38.7226231
time: 0.55 seconds

## Relational analysis of NS_B1_A1_A2_B2

### Relational analysis result of NS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7489006, upper bound: 38.7650966
time: 0.60 seconds

## BFS NS instance: NS_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.4001608, 16.3969440, -3.2743556, 12.2697954, -16.6699524, 19.6712990
1: -12.4139957, 33.8227081, -9.2661772, 25.2540913, -37.6680870, 43.0888863
2: -19.1561890, 31.2012348, -14.2485094, 23.2573910, -42.4135666, 45.4497414
3: -10.3882036, 39.4758148, -7.7333012, 29.5679417, -39.9561310, 47.2091141
4: -17.2259312, 27.3772354, -12.8016291, 20.4090157, -37.6349449, 40.1788597

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_B1_A1

### Relational analysis result of NS_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7198826, upper bound: 38.7411639
time: 0.85 seconds

## Relational analysis of NS_B1_A2_B1_A2

### Relational analysis result of NS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7648562, upper bound: 38.7516429
time: 0.73 seconds

## BFS NS instance: NS_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.4397011, 16.5434704, -3.6858134, 13.7642956, -18.2039948, 20.2292843
1: -12.5246544, 34.1262398, -10.3912401, 28.3749695, -40.8996201, 44.5174789
2: -19.3276100, 31.4766617, -16.0099182, 26.1017380, -45.4293480, 47.4865799
3: -10.4815588, 39.8213959, -8.6881809, 33.1576347, -43.6391945, 48.5095673
4: -17.3803291, 27.6185970, -14.3799839, 22.8897057, -40.2700310, 41.9985809

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_B2_A1

### Relational analysis result of NS_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7503020, upper bound: 38.7567658
time: 0.76 seconds

## Relational analysis of NS_B1_A2_B2_A2

### Relational analysis result of NS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7668775, upper bound: 38.7652505
time: 1.01 seconds

## BFS NS instance: NS_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3.7588351, 14.0336018, -4.4040232, 16.4103699, -20.1692047, 18.4376240
1: -10.6033792, 28.9314671, -12.4216051, 33.8487663, -44.4521446, 41.3530693
2: -16.3532505, 26.6089439, -19.1634502, 31.2314987, -47.5847473, 45.7723923
3: -8.8667536, 33.7970123, -10.3947983, 39.5014305, -48.3681793, 44.1918068
4: -14.6861916, 23.3346539, -17.2335968, 27.4032249, -42.0894165, 40.5682526

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_B2_A1

### Relational analysis result of NS_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7224665, upper bound: 38.7485574
time: 0.57 seconds

## Relational analysis of NS_B2_A1_B2_A2

### Relational analysis result of NS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7224665, upper bound: 38.7668775
time: 0.66 seconds

## BFS NS instance: NS_B2_A2_A1

### Backsubstitution after applying NS history:
0: -3.3036442, 12.3235006, -4.2705116, 15.9111643, -19.2148056, 16.5940132
1: -9.2921839, 25.4433823, -12.0345850, 32.8220978, -42.1142807, 37.4779663
2: -14.2929726, 23.3519344, -18.5508537, 30.2805557, -44.5735207, 41.9027863
3: -7.7610064, 29.7456169, -10.0700426, 38.3095093, -46.0705147, 39.8156586
4: -12.8320026, 20.4928341, -16.6836739, 26.5694542, -39.4014587, 37.1765060

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_A1_B1

### Relational analysis result of NS_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7493014, upper bound: 38.7493014
time: 0.63 seconds

## Relational analysis of NS_B2_A2_A1_B2

### Relational analysis result of NS_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7493014, upper bound: 38.7493014
time: 0.77 seconds

## BFS NS instance: NS_B2_A2_A2

### Backsubstitution after applying NS history:
0: -4.4040232, 16.4103699, -4.4357328, 16.5286694, -20.9326916, 20.8460999
1: -12.4216051, 33.8487663, -12.5131998, 34.0953102, -46.5169106, 46.3619614
2: -19.1634502, 31.2314987, -19.3093472, 31.4494019, -50.6128540, 50.5408478
3: -10.3947983, 39.5014305, -10.4719114, 39.7858009, -50.1805954, 49.9733429
4: -17.2335968, 27.4032249, -17.3640079, 27.5946503, -44.8282471, 44.7672310

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_A2_B1

### Relational analysis result of NS_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7597315, upper bound: 38.7493014
time: 0.66 seconds

## Relational analysis of NS_B2_A2_A2_B2

### Relational analysis result of NS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7597315, upper bound: 38.7670314
time: 1.29 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.84 seconds
NS_B1_A1_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.84
Output dim: 4, lower bound: -38.7489006, upper bound: 38.7226231
NS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -38.7489006, upper bound: 38.7650966
NS_B1_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 2.84
Output dim: 4, lower bound: -38.7198826, upper bound: 38.7411639
NS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -38.7648562, upper bound: 38.7516429
NS_B1_A2_B2_A1, status: Status.VERIFIED, split count: 4, time: 2.84
Output dim: 4, lower bound: -38.7503020, upper bound: 38.7567658
NS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -38.7668775, upper bound: 38.7652505
NS_B2_A1_B2_A1, status: Status.VERIFIED, split count: 4, time: 2.84
Output dim: 4, lower bound: -38.7224665, upper bound: 38.7485574
NS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -38.7224665, upper bound: 38.7668775
NS_B2_A2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.84
Output dim: 4, lower bound: -38.7493014, upper bound: 38.7493014
NS_B2_A2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.84
Output dim: 4, lower bound: -38.7493014, upper bound: 38.7493014
NS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -38.7597315, upper bound: 38.7493014
NS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -38.7597315, upper bound: 38.7670314

## BFS NS instance: NS_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -3.7267268, 13.9167728, -3.7267268, 13.9167728, -17.6434956, 17.6434975
1: -10.5112848, 28.6832466, -10.5112848, 28.6832466, -39.1945267, 39.1945229
2: -16.2075958, 26.3915825, -16.2075958, 26.3915825, -42.5991783, 42.5991783
3: -8.7892141, 33.5076637, -8.7892141, 33.5076637, -42.2968712, 42.2968750
4: -14.5561237, 23.1437645, -14.5561237, 23.1437645, -37.6998901, 37.6998863

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_A1_A2_B2_A1

### Relational analysis result of NS_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7467593, upper bound: 38.7626262
time: 0.91 seconds

## Relational analysis of NS_B1_A1_A2_B2_A2

### Relational analysis result of NS_B1_A1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7470022, upper bound: 38.7205540
time: 0.82 seconds

## BFS NS instance: NS_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.3642921, 16.2633228, -3.2704887, 12.2555485, -16.6198368, 19.5338058
1: -12.3104849, 33.5442581, -9.2550392, 25.2243423, -37.5348091, 42.7992973
2: -18.9913445, 30.9551601, -14.2308102, 23.2306652, -42.2219963, 45.1859703
3: -10.3010607, 39.1542778, -7.7239251, 29.5339355, -39.8349953, 46.8782043
4: -17.0786991, 27.1608353, -12.7857780, 20.3855419, -37.4642410, 39.9466095

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7405415, upper bound: 38.7072749
time: 0.52 seconds

## Relational analysis of NS_B1_A2_B1_A2_B2

### Relational analysis result of NS_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7405415, upper bound: 38.7516429
time: 0.64 seconds

## BFS NS instance: NS_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.4040232, 16.4103699, -3.6813405, 13.7480240, -18.1520462, 20.0917053
1: -12.4216051, 33.8487663, -10.3783598, 28.3406773, -40.7622719, 44.2271271
2: -19.1634502, 31.2314987, -15.9898615, 26.0715542, -45.2350044, 47.2213593
3: -10.3947983, 39.5014305, -8.6773958, 33.1180763, -43.5128746, 48.1788216
4: -17.2335968, 27.4032249, -14.3620787, 22.8632717, -40.0968628, 41.7653046

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_B2_A2_B1

### Relational analysis result of NS_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7481839, upper bound: 38.7218277
time: 0.63 seconds

## Relational analysis of NS_B1_A2_B2_A2_B2

### Relational analysis result of NS_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7481839, upper bound: 38.7218277
time: 0.83 seconds

## BFS NS instance: NS_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.7267268, 13.9167728, -4.4040232, 16.4103699, -20.1370945, 18.3207912
1: -10.5112848, 28.6832466, -12.4216051, 33.8487663, -44.3600502, 41.1048393
2: -16.2075958, 26.3915825, -19.1634502, 31.2314987, -47.4390945, 45.5550308
3: -8.7892141, 33.5076637, -10.3947983, 39.5014305, -48.2906342, 43.9024582
4: -14.5561237, 23.1437645, -17.2335968, 27.4032249, -41.9593506, 40.3773613

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B2_A1_B2_A2_B1

### Relational analysis result of NS_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.6654498, upper bound: 38.7668526
time: 0.59 seconds

## Relational analysis of NS_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 18

## BFS NS instance: NS_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -4.4040232, 16.4103699, -3.3036442, 12.3235006, -16.7275238, 19.7140102
1: -12.4216051, 33.8487663, -9.2921839, 25.4433823, -37.8649864, 43.1409492
2: -19.1634502, 31.2314987, -14.2929726, 23.3519344, -42.5153847, 45.5244675
3: -10.3947983, 39.5014305, -7.7610064, 29.7456169, -40.1404152, 47.2624283
4: -17.2335968, 27.4032249, -12.8320026, 20.4928341, -37.7264328, 40.2352295

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B2_A2_A2_B1_B1

### Relational analysis result of NS_B2_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7444270, upper bound: 38.7446072
time: 0.90 seconds

## Relational analysis of NS_B2_A2_A2_B1_B2

### Relational analysis result of NS_B2_A2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7591104, upper bound: 38.7447678
time: 1.02 seconds

## BFS NS instance: NS_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -4.4040232, 16.4103699, -4.4040232, 16.4103699, -20.8143902, 20.8143883
1: -12.4216051, 33.8487663, -12.4216051, 33.8487663, -46.2703667, 46.2703667
2: -19.1634502, 31.2314987, -19.1634502, 31.2314987, -50.3949509, 50.3949509
3: -10.3947983, 39.5014305, -10.3947983, 39.5014305, -49.8962250, 49.8962250
4: -17.2335968, 27.4032249, -17.2335968, 27.4032249, -44.6368217, 44.6368217

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B2_A2_A2_B2_A1

### Relational analysis result of NS_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7593532, upper bound: 38.7637179
time: 0.76 seconds

## Relational analysis of NS_B2_A2_A2_B2_A2

### Relational analysis result of NS_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7591104, upper bound: 38.7670190
time: 0.72 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.45 seconds
NS_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 4, lower bound: -38.7467593, upper bound: 38.7626262
NS_B1_A1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.45
Output dim: 4, lower bound: -38.7470022, upper bound: 38.7205540
NS_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 5, time: 2.45
Output dim: 4, lower bound: -38.7405415, upper bound: 38.7072749
NS_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 5, time: 2.45
Output dim: 4, lower bound: -38.7405415, upper bound: 38.7516429
NS_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 2.45
Output dim: 4, lower bound: -38.7481839, upper bound: 38.7218277
NS_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 2.45
Output dim: 4, lower bound: -38.7481839, upper bound: 38.7218277
NS_B2_A2_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 2.45
Output dim: 4, lower bound: -38.7444270, upper bound: 38.7446072
NS_B2_A2_A2_B1_B2, status: Status.VERIFIED, split count: 5, time: 2.45
Output dim: 4, lower bound: -38.7591104, upper bound: 38.7447678
NS_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 4, lower bound: -38.7593532, upper bound: 38.7637179
NS_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 4, lower bound: -38.7591104, upper bound: 38.7670190

## BFS NS instance: NS_B1_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.5876076, 13.4000835, -3.7267268, 13.9167728, -17.5043774, 17.1268101
1: -10.1229544, 27.6234550, -10.5112848, 28.6832466, -38.8061905, 38.1347389
2: -15.6257172, 25.3450069, -16.2075958, 26.3915825, -42.0172958, 41.5526047
3: -8.4637814, 32.2602921, -8.7892141, 33.5076637, -41.9714432, 41.0494957
4: -14.0247498, 22.2334137, -14.5561237, 23.1437645, -37.1685143, 36.7895355

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B1_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B1_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B1_A1_A2_B2_A1_B1

### Relational analysis result of NS_B1_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7625503, upper bound: 38.7625616
time: 0.70 seconds

## Relational analysis of NS_B1_A1_A2_B2_A1_B2

### Relational analysis result of NS_B1_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7625503, upper bound: 38.7625616
time: 0.74 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.0519195, 15.1149302, -4.3569274, 16.2367764, -20.2886963, 19.4718571
1: -11.4453869, 31.1698246, -12.2895346, 33.4903679, -44.9357491, 43.4593582
2: -17.6947441, 28.6487045, -18.9637051, 30.8812084, -48.5759315, 47.6123962
3: -9.5782471, 36.3911858, -10.2844782, 39.0823746, -48.6606216, 46.6756630
4: -15.8977432, 25.1365299, -17.0517025, 27.0955124, -42.9932518, 42.1882324

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B2_A2_A2_B2_A1_B1

### Relational analysis result of NS_B2_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7636826, upper bound: 38.7636826
time: 0.91 seconds

## Relational analysis of NS_B2_A2_A2_B2_A1_B2

### Relational analysis result of NS_B2_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7636826, upper bound: 38.7637179
time: 0.71 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.3185067, 16.0895386, -4.4040232, 16.4103699, -20.7288742, 20.4935551
1: -12.1804075, 33.1850700, -12.4216051, 33.8487663, -46.0291748, 45.6066742
2: -18.8059826, 30.5994816, -19.1634502, 31.2314987, -50.0374794, 49.7629318
3: -10.1931973, 38.7432060, -10.3947983, 39.5014305, -49.6946259, 49.1379967
4: -16.9071827, 26.8485527, -17.2335968, 27.4032249, -44.3104095, 44.0821495

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B2_A2_A2_B2_A2_B1

### Relational analysis result of NS_B2_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7637209, upper bound: 38.7636826
time: 0.78 seconds

## Relational analysis of NS_B2_A2_A2_B2_A2_B2

### Relational analysis result of NS_B2_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7637209, upper bound: 38.7670190
time: 0.91 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.78 seconds
NS_B1_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 4, lower bound: -38.7625503, upper bound: 38.7625616
NS_B1_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 4, lower bound: -38.7625503, upper bound: 38.7625616
NS_B2_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 4, lower bound: -38.7636826, upper bound: 38.7636826
NS_B2_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 4, lower bound: -38.7636826, upper bound: 38.7637179
NS_B2_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 4, lower bound: -38.7637209, upper bound: 38.7636826
NS_B2_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.78
Output dim: 4, lower bound: -38.7637209, upper bound: 38.7670190

## BFS NS instance: NS_B1_A1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -3.5876076, 13.4000835, -3.5876076, 13.4000835, -16.9876919, 16.9876919
1: -10.1229544, 27.6234550, -10.1229544, 27.6234550, -37.7464066, 37.7464066
2: -15.6257172, 25.3450069, -15.6257172, 25.3450069, -40.9707222, 40.9707222
3: -8.4637814, 32.2602921, -8.4637814, 32.2602921, -40.7240715, 40.7240715
4: -14.0247498, 22.2334137, -14.0247498, 22.2334137, -36.2581635, 36.2581635

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B1

### Relational analysis result of NS_B1_A1_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7614403, upper bound: 38.7504394
time: 0.75 seconds

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B2

### Relational analysis result of NS_B1_A1_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7631236, upper bound: 38.7622143
time: 1.02 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3.5876076, 13.4000835, -3.4762747, 12.9826984, -16.5703068, 16.8763580
1: -10.1229544, 27.6234550, -9.8169394, 26.7767887, -36.8997421, 37.4403915
2: -15.6257172, 25.3450069, -15.1452837, 24.5250301, -40.1507301, 40.4902916
3: -8.4637814, 32.2602921, -8.2069397, 31.2991447, -39.7629242, 40.4672279
4: -14.0247498, 22.2334137, -13.5864372, 21.5058670, -35.5306168, 35.8198509

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_A2_B2_A1_B2_B1

### Relational analysis result of NS_B1_A1_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7614403, upper bound: 38.7504394
time: 0.69 seconds

## Relational analysis of NS_B1_A1_A2_B2_A1_B2_B2

### Relational analysis result of NS_B1_A1_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7631236, upper bound: 38.7622143
time: 1.09 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -4.0519195, 15.1149302, -4.0519195, 15.1149302, -19.1668491, 19.1668491
1: -11.4453869, 31.1698246, -11.4453869, 31.1698246, -42.6152115, 42.6152115
2: -17.6947441, 28.6487045, -17.6947441, 28.6487045, -46.3434372, 46.3434334
3: -9.5782471, 36.3911858, -9.5782471, 36.3911858, -45.9694328, 45.9694328
4: -15.8977432, 25.1365299, -15.8977432, 25.1365299, -41.0342712, 41.0342712

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A1

### Relational analysis result of NS_B2_A2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7631710, upper bound: 38.7623127
time: 0.69 seconds

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A2

### Relational analysis result of NS_B2_A2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7622838, upper bound: 38.7622838
time: 0.73 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -4.0519195, 15.1149302, -4.3185067, 16.0895386, -20.1414547, 19.4334373
1: -11.4453869, 31.1698246, -12.1804075, 33.1850700, -44.6304550, 43.3502312
2: -17.6947441, 28.6487045, -18.8059826, 30.5994816, -48.2942085, 47.4546814
3: -9.5782471, 36.3911858, -10.1931973, 38.7432060, -48.3214531, 46.5843811
4: -15.8977432, 25.1365299, -16.9071827, 26.8485527, -42.7462921, 42.0437126

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A1

### Relational analysis result of NS_B2_A2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7631710, upper bound: 38.7635862
time: 0.71 seconds

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A2

### Relational analysis result of NS_B2_A2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7622838, upper bound: 38.7634138
time: 0.72 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.3185067, 16.0895386, -4.0519195, 15.1149302, -19.4334373, 20.1414547
1: -12.1804075, 33.1850700, -11.4453869, 31.1698246, -43.3502312, 44.6304550
2: -18.8059826, 30.5994816, -17.6947441, 28.6487045, -47.4546852, 48.2942085
3: -10.1931973, 38.7432060, -9.5782471, 36.3911858, -46.5843811, 48.3214531
4: -16.9071827, 26.8485527, -15.8977432, 25.1365299, -42.0437126, 42.7462921

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_A2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B2_A2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_A2_B2_A2_B1_A1

### Relational analysis result of NS_B2_A2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7439766, upper bound: 38.7460001
time: 0.83 seconds

## Relational analysis of NS_B2_A2_A2_B2_A2_B1_A2

### Relational analysis result of NS_B2_A2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7622096, upper bound: 38.7621651
time: 0.62 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.3185067, 16.0895386, -4.3185067, 16.0895386, -20.4080410, 20.4080410
1: -12.1804075, 33.1850700, -12.1804075, 33.1850700, -45.3654785, 45.3654785
2: -18.8059826, 30.5994816, -18.8059826, 30.5994816, -49.4054565, 49.4054604
3: -10.1931973, 38.7432060, -10.1931973, 38.7432060, -48.9363976, 48.9363976
4: -16.9071827, 26.8485527, -16.9071827, 26.8485527, -43.7557373, 43.7557373

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_A2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B2_A2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_A2_B2_A2_B2_B1

### Relational analysis result of NS_B2_A2_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7441266, upper bound: 38.7506993
time: 0.68 seconds

## Relational analysis of NS_B2_A2_A2_B2_A2_B2_B2

### Relational analysis result of NS_B2_A2_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7622096, upper bound: 38.7661261
time: 0.76 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 4.21 seconds
NS_B1_A1_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 4, lower bound: -38.7614403, upper bound: 38.7504394
NS_B1_A1_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 4, lower bound: -38.7631236, upper bound: 38.7622143
NS_B1_A1_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 4, lower bound: -38.7614403, upper bound: 38.7504394
NS_B1_A1_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 4, lower bound: -38.7631236, upper bound: 38.7622143
NS_B2_A2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 4, lower bound: -38.7631710, upper bound: 38.7623127
NS_B2_A2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 4, lower bound: -38.7622838, upper bound: 38.7622838
NS_B2_A2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 4, lower bound: -38.7631710, upper bound: 38.7635862
NS_B2_A2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 4, lower bound: -38.7622838, upper bound: 38.7634138
NS_B2_A2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.21
Output dim: 4, lower bound: -38.7439766, upper bound: 38.7460001
NS_B2_A2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 4, lower bound: -38.7622096, upper bound: 38.7621651
NS_B2_A2_A2_B2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.21
Output dim: 4, lower bound: -38.7441266, upper bound: 38.7506993
NS_B2_A2_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 4, lower bound: -38.7622096, upper bound: 38.7661261

## BFS NS instance: NS_B1_A1_A2_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -3.5876076, 13.4000835, -3.3237505, 12.4452143, -16.0328217, 16.7238331
1: -10.1229544, 27.6234550, -9.3699265, 25.6169167, -35.7398720, 36.9933777
2: -15.6257172, 25.3450069, -14.4312811, 23.5223274, -39.1480408, 39.7762871
3: -8.4637814, 32.2602921, -7.8283730, 29.9616127, -38.4253883, 40.0886574
4: -14.0247498, 22.2334137, -12.9475021, 20.6309223, -34.6556702, 35.1809158

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B1_A1

### Relational analysis result of NS_B1_A1_A2_B2_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7490958, upper bound: 38.7490959
time: 0.75 seconds

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B1_A1

### Relational analysis result of NS_B1_A1_A2_B2_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7569484, upper bound: 38.7479537
time: 1.08 seconds

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B1_A2

### Relational analysis result of NS_B1_A1_A2_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7610353, upper bound: 38.7485149
time: 0.72 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -3.5560558, 13.2835751, -3.4060473, 12.7292948, -16.2853508, 16.6896229
1: -10.0332861, 27.3805809, -9.6026545, 26.2345314, -36.2678032, 36.9832344
2: -15.4874706, 25.1254063, -14.8039293, 24.1747532, -39.6622238, 39.9293365
3: -8.3885956, 31.9778576, -8.0260477, 30.6547241, -39.0433006, 40.0039062
4: -13.9002142, 22.0402966, -13.2851677, 21.1903687, -35.0905838, 35.3254623

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B2_B1

### Relational analysis result of NS_B1_A1_A2_B2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7624641, upper bound: 38.7627477
time: 1.01 seconds

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B2_B2

### Relational analysis result of NS_B1_A1_A2_B2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7624641, upper bound: 38.7624640
time: 0.76 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -3.5876076, 13.4000835, -3.2110589, 12.0169401, -15.6045456, 16.6111431
1: -10.1229544, 27.6234550, -9.0590935, 24.7582493, -34.8812027, 36.6825485
2: -15.6257172, 25.3450069, -13.9393806, 22.6828575, -38.3085747, 39.2843857
3: -8.4637814, 32.2602921, -7.5665579, 28.9855404, -37.4493217, 39.8268509
4: -14.0247498, 22.2334137, -12.4995861, 19.8945618, -33.9193077, 34.7330017

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B1_A1_A2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_A2_B2_A1_B2_B1_A1

### Relational analysis result of NS_B1_A1_A2_B2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7486500, upper bound: 38.7486500
time: 0.62 seconds

## Relational analysis of NS_B1_A1_A2_B2_A1_B2_B1_A2

### Relational analysis result of NS_B1_A1_A2_B2_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7486500, upper bound: 38.7504394
time: 0.78 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -3.5560558, 13.2835751, -3.3233368, 12.4083958, -15.9644489, 16.6069126
1: -10.0332861, 27.3805809, -9.3790207, 25.5938931, -35.6271706, 36.7596016
2: -15.4874706, 25.1254063, -14.4472017, 23.5540066, -39.0414772, 39.5726013
3: -8.3885956, 31.9778576, -7.8380136, 29.9325752, -38.3211517, 39.8158722
4: -13.9002142, 22.0402966, -12.9625778, 20.6496468, -34.5498581, 35.0028763

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B1_A1_A2_B2_A1_B2_B2_B1

### Relational analysis result of NS_B1_A1_A2_B2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7613300, upper bound: 38.7614090
time: 0.64 seconds

## Relational analysis of NS_B1_A1_A2_B2_A1_B2_B2_B2

### Relational analysis result of NS_B1_A1_A2_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7623556, upper bound: 38.7613300
time: 0.70 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -4.0256953, 15.0191994, -4.0519195, 15.1149302, -19.1406250, 19.0711155
1: -11.3737621, 30.9658165, -11.4453869, 31.1698246, -42.5435867, 42.4112015
2: -17.5865784, 28.4613533, -17.6947441, 28.6487045, -46.2352676, 46.1560898
3: -9.5185661, 36.1559486, -9.5782471, 36.3911858, -45.9097519, 45.7341957
4: -15.7998934, 24.9679375, -15.8977432, 25.1365299, -40.9364243, 40.8656693

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_B2_A2_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7548578, upper bound: 38.7463164
time: 0.60 seconds

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_B2_A2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7615598, upper bound: 38.7606987
time: 0.80 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4.0319285, 15.0403442, -4.0519195, 15.1149302, -19.1468582, 19.0922642
1: -11.3893051, 31.0175285, -11.4453869, 31.1698246, -42.5591240, 42.4629135
2: -17.6096230, 28.5012341, -17.6947441, 28.6487045, -46.2583199, 46.1959610
3: -9.5310717, 36.2150803, -9.5782471, 36.3911858, -45.9222565, 45.7933273
4: -15.8199406, 25.0073395, -15.8977432, 25.1365299, -40.9564705, 40.9050827

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_B2_A2_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7525736, upper bound: 38.7461266
time: 0.75 seconds

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_B2_A2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7606717, upper bound: 38.7606716
time: 0.68 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -4.0256953, 15.0191994, -4.3185067, 16.0895386, -20.1152306, 19.3377037
1: -11.3737621, 30.9658165, -12.1804075, 33.1850700, -44.5588303, 43.1462250
2: -17.5865784, 28.4613533, -18.8059826, 30.5994816, -48.1860504, 47.2673340
3: -9.5185661, 36.1559486, -10.1931973, 38.7432060, -48.2617683, 46.3491440
4: -15.7998934, 24.9679375, -16.9071827, 26.8485527, -42.6484413, 41.8751221

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_B2_A2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7605408, upper bound: 38.7521181
time: 0.89 seconds

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_B2_A2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7620860, upper bound: 38.7618939
time: 0.77 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4.0319285, 15.0403442, -4.3185067, 16.0895386, -20.1214619, 19.3588505
1: -11.3893051, 31.0175285, -12.1804075, 33.1850700, -44.5743713, 43.1979370
2: -17.6096230, 28.5012341, -18.8059826, 30.5994816, -48.2090950, 47.3072090
3: -9.5310717, 36.2150803, -10.1931973, 38.7432060, -48.2742729, 46.4082756
4: -15.8199406, 25.0073395, -16.9071827, 26.8485527, -42.6684952, 41.9145203

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_B2_A2_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7591671, upper bound: 38.7519936
time: 0.73 seconds

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_B2_A2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7607097, upper bound: 38.7616877
time: 0.88 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.1050639, 15.3071823, -4.0184064, 14.9903526, -19.0954170, 19.3255825
1: -11.5664978, 31.5826912, -11.3502417, 30.9140358, -42.4805298, 42.9329300
2: -17.8315830, 29.1853523, -17.5472221, 28.4122906, -46.2438698, 46.7325745
3: -9.6777325, 36.9075699, -9.4983673, 36.0945435, -45.7722778, 46.4059372
4: -16.0339794, 25.6013775, -15.7646360, 24.9288158, -40.9627953, 41.3660126

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B2_A2_A2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_B2_A2_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7441263, upper bound: 38.7445584
time: 0.60 seconds

## Relational analysis of NS_B2_A2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_B2_A2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7441266, upper bound: 38.7621651
time: 0.70 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -4.2843742, 15.9625645, -4.1050639, 15.3071823, -19.5915565, 20.0676289
1: -12.0834875, 32.9238968, -11.5664978, 31.5826912, -43.6661720, 44.4903946
2: -18.6558762, 30.3588390, -17.8315830, 29.1853523, -47.8412247, 48.1904182
3: -10.1118202, 38.4398804, -9.6777325, 36.9075699, -47.0193901, 48.1176147
4: -16.7717037, 26.6370869, -16.0339794, 25.6013775, -42.3730812, 42.6710663

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B2_A2_A2_B2_A2_B2_B2_B1

### Relational analysis result of NS_B2_A2_A2_B2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7652359, upper bound: 38.7652359
time: 0.64 seconds

## Relational analysis of NS_B2_A2_A2_B2_A2_B2_B2_B2

### Relational analysis result of NS_B2_A2_A2_B2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7657876, upper bound: 38.7657875
time: 0.70 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 2.29 seconds
NS_B1_A1_A2_B2_A1_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 2.29
Output dim: 4, lower bound: -38.7569484, upper bound: 38.7479537
NS_B1_A1_A2_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 4, lower bound: -38.7610353, upper bound: 38.7485149
NS_B1_A1_A2_B2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 4, lower bound: -38.7624641, upper bound: 38.7627477
NS_B1_A1_A2_B2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 4, lower bound: -38.7624641, upper bound: 38.7624640
NS_B1_A1_A2_B2_A1_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 2.29
Output dim: 4, lower bound: -38.7486500, upper bound: 38.7486500
NS_B1_A1_A2_B2_A1_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 2.29
Output dim: 4, lower bound: -38.7486500, upper bound: 38.7504394
NS_B1_A1_A2_B2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 4, lower bound: -38.7613300, upper bound: 38.7614090
NS_B1_A1_A2_B2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 4, lower bound: -38.7623556, upper bound: 38.7613300
NS_B2_A2_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.29
Output dim: 4, lower bound: -38.7548578, upper bound: 38.7463164
NS_B2_A2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 4, lower bound: -38.7615598, upper bound: 38.7606987
NS_B2_A2_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.29
Output dim: 4, lower bound: -38.7525736, upper bound: 38.7461266
NS_B2_A2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 4, lower bound: -38.7606717, upper bound: 38.7606716
NS_B2_A2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 4, lower bound: -38.7605408, upper bound: 38.7521181
NS_B2_A2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 4, lower bound: -38.7620860, upper bound: 38.7618939
NS_B2_A2_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.29
Output dim: 4, lower bound: -38.7591671, upper bound: 38.7519936
NS_B2_A2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 4, lower bound: -38.7607097, upper bound: 38.7616877
NS_B2_A2_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.29
Output dim: 4, lower bound: -38.7441263, upper bound: 38.7445584
NS_B2_A2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 4, lower bound: -38.7441266, upper bound: 38.7621651
NS_B2_A2_A2_B2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 4, lower bound: -38.7652359, upper bound: 38.7652359
NS_B2_A2_A2_B2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 4, lower bound: -38.7657876, upper bound: 38.7657875

## BFS NS instance: NS_B1_A1_A2_B2_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -3.6236544, 13.5335817, -3.3237505, 12.4452143, -16.0688686, 16.8573284
1: -10.2137003, 27.8816967, -9.3699265, 25.6169167, -35.8306122, 37.2516136
2: -15.7433825, 25.6307869, -14.4312811, 23.5223274, -39.2657089, 40.0620689
3: -8.5409508, 32.5596924, -7.8283730, 29.9616127, -38.5025597, 40.3880615
4: -14.1373873, 22.4841290, -12.9475021, 20.6309223, -34.7683105, 35.4316330

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 32

## BFS NS instance: NS_B1_A1_A2_B2_A1_B1_B2_B1

### Backsubstitution after applying NS history:
0: -3.5345473, 13.2025394, -3.3846812, 12.6328506, -16.1673985, 16.5872211
1: -9.9729586, 27.2146454, -9.5247660, 26.0384464, -36.0114059, 36.7394104
2: -15.3963966, 24.9673615, -14.6906567, 23.9218197, -39.3182144, 39.6580200
3: -8.3379288, 31.7852192, -7.9672227, 30.4239807, -38.7619095, 39.7524338
4: -13.8174362, 21.9019775, -13.1828041, 20.9925900, -34.8100166, 35.0847740

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B2_B1_A1

### Relational analysis result of NS_B1_A1_A2_B2_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7624641, upper bound: 38.7624640
time: 0.56 seconds

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B2_B1_A2

### Relational analysis result of NS_B1_A1_A2_B2_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7624641, upper bound: 38.7624640
time: 1.17 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A1_B1_B2_B2

### Backsubstitution after applying NS history:
0: -3.5560558, 13.2835751, -3.3880939, 12.6577854, -16.2138405, 16.6716690
1: -10.0332861, 27.3805809, -9.5504379, 26.0920467, -36.1253281, 36.9310188
2: -15.4874706, 25.1254063, -14.7228165, 24.0534801, -39.5409508, 39.8482170
3: -8.3885956, 31.9778576, -7.9822702, 30.4840355, -38.8726196, 39.9601212
4: -13.9002142, 22.0402966, -13.2119770, 21.0844860, -34.9846992, 35.2522736

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B2_B2_A1

### Relational analysis result of NS_B1_A1_A2_B2_A1_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7585925, upper bound: 38.7586622
time: 0.66 seconds

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B2_B2_A2

### Relational analysis result of NS_B1_A1_A2_B2_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7624622, upper bound: 38.7624622
time: 0.66 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -3.5345473, 13.2025394, -3.3930862, 12.6570673, -16.1916142, 16.5956249
1: -9.9729586, 27.2146454, -9.5597191, 26.1214619, -36.0944214, 36.7743645
2: -15.3963966, 24.9673615, -14.7239027, 23.9441566, -39.3405533, 39.6912651
3: -8.3379288, 31.7852192, -7.9910865, 30.5341930, -38.8721161, 39.7762909
4: -13.8174362, 21.9019775, -13.2074881, 21.0102291, -34.8276634, 35.1094627

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B1_A1_A2_B2_A1_B2_B2_B1_A1

### Relational analysis result of NS_B1_A1_A2_B2_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7613300, upper bound: 38.7613300
time: 0.79 seconds

## Relational analysis of NS_B1_A1_A2_B2_A1_B2_B2_B1_A2

### Relational analysis result of NS_B1_A1_A2_B2_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7613300, upper bound: 38.7613300
time: 0.69 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -3.5560558, 13.2835751, -3.3053248, 12.3360424, -15.8920975, 16.5888996
1: -10.0332861, 27.3805809, -9.3266153, 25.4506264, -35.4839020, 36.7071953
2: -15.4874706, 25.1254063, -14.3663054, 23.4312305, -38.9187012, 39.4917107
3: -8.3885956, 31.9778576, -7.7941871, 29.7599602, -38.1485443, 39.7720451
4: -13.9002142, 22.0402966, -12.8895359, 20.5422935, -34.4425049, 34.9298325

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_A2_B2_A1_B2_B2_B2_A1

### Relational analysis result of NS_B1_A1_A2_B2_A1_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7486500, upper bound: 38.7516436
time: 0.72 seconds

## Relational analysis of NS_B1_A1_A2_B2_A1_B2_B2_B2_A2

### Relational analysis result of NS_B1_A1_A2_B2_A1_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7486500, upper bound: 38.7486500
time: 1.35 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3.9940517, 14.9016790, -3.8419523, 14.3325024, -18.3265533, 18.7436314
1: -11.2842083, 30.7247086, -10.8419476, 29.5634613, -40.8476601, 41.5666580
2: -17.4480000, 28.2377701, -16.7325363, 27.2673683, -44.7153702, 44.9703064
3: -9.4433794, 35.8763275, -9.0709143, 34.5527687, -43.9961472, 44.9472427
4: -15.6747637, 24.7712574, -15.0358334, 23.9179802, -39.5927391, 39.8070908

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A1_B2_B1

### Relational analysis result of NS_B2_A2_A2_B2_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7445396, upper bound: 38.7606986
time: 0.81 seconds

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A1_B2_B2

### Relational analysis result of NS_B2_A2_A2_B2_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7615598, upper bound: 38.7606987
time: 0.74 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3.9985857, 14.9164171, -3.8419523, 14.3325024, -18.3310890, 18.7583694
1: -11.2946062, 30.7630329, -10.8419476, 29.5634613, -40.8580666, 41.6049767
2: -17.4627781, 28.2661839, -16.7325363, 27.2673683, -44.7301369, 44.9987183
3: -9.4515896, 35.9199409, -9.0709143, 34.5527687, -44.0043564, 44.9908562
4: -15.6874590, 24.8008060, -15.0358334, 23.9179802, -39.6054382, 39.8366356

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_B2_A2_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7439694, upper bound: 38.7443414
time: 0.84 seconds

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_B2_A2_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7439694, upper bound: 38.7606717
time: 0.72 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -4.0256953, 15.0191994, -4.0459719, 15.1032467, -19.1289406, 19.0651703
1: -11.3737621, 30.9658165, -11.3990774, 31.1168232, -42.4905853, 42.3648949
2: -17.5865784, 28.4613533, -17.5763721, 28.7252522, -46.3118172, 46.0377274
3: -9.5185661, 36.1559486, -9.5364075, 36.3867760, -45.9053421, 45.6923561
4: -15.7998934, 24.9679375, -15.7984543, 25.2109222, -41.0108147, 40.7663879

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 32

## BFS NS instance: NS_B2_A2_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3.9940517, 14.9016790, -4.1050639, 15.3071823, -19.3012314, 19.0067425
1: -11.2842083, 30.7247086, -11.5664978, 31.5826912, -42.8668938, 42.2912064
2: -17.4480000, 28.2377701, -17.8315830, 29.1853523, -46.6333504, 46.0693512
3: -9.4433794, 35.8763275, -9.6777325, 36.9075699, -46.3509483, 45.5540619
4: -15.6747637, 24.7712574, -16.0339794, 25.6013775, -41.2761345, 40.8052368

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A1_B2_B1

### Relational analysis result of NS_B2_A2_A2_B2_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7620860, upper bound: 38.7618939
time: 0.80 seconds

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A1_B2_B2

### Relational analysis result of NS_B2_A2_A2_B2_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7620860, upper bound: 38.7618939
time: 0.67 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3.9985857, 14.9164171, -4.1050639, 15.3071823, -19.3057632, 19.0214806
1: -11.2946062, 30.7630329, -11.5664978, 31.5826912, -42.8772964, 42.3295288
2: -17.4627781, 28.2661839, -17.8315830, 29.1853523, -46.6481133, 46.0977592
3: -9.4515896, 35.9199409, -9.6777325, 36.9075699, -46.3591614, 45.5976715
4: -15.6874590, 24.8008060, -16.0339794, 25.6013775, -41.2888336, 40.8347816

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_B2_A2_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7439694, upper bound: 38.7441529
time: 0.61 seconds

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 32

## BFS NS instance: NS_B2_A2_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.1050639, 15.3071823, -3.8419523, 14.3325024, -18.4375668, 19.1491337
1: -11.5664978, 31.5826912, -10.8419476, 29.5634613, -41.1299591, 42.4246368
2: -17.8315830, 29.1853523, -16.7325363, 27.2673683, -45.0989494, 45.9178848
3: -9.6777325, 36.9075699, -9.0709143, 34.5527687, -44.2304993, 45.9784851
4: -16.0339794, 25.6013775, -15.0358334, 23.9179802, -39.9519577, 40.6372070

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_A2_A2_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B2_A2_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_A2_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## BFS NS instance: NS_B2_A2_A2_B2_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -4.2624426, 15.8805141, -4.1591334, 15.4848976, -19.7473354, 20.0396461
1: -12.0220604, 32.7572098, -11.7106962, 31.9519043, -43.9739647, 44.4679070
2: -18.5630512, 30.1918659, -18.0774899, 29.4612045, -48.0242500, 48.2693520
3: -10.0599852, 38.2369232, -9.8021688, 37.2724075, -47.3323936, 48.0390854
4: -16.6866951, 26.4913139, -16.2457333, 25.8559608, -42.5426559, 42.7370453

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B2_A2_A2_B2_A2_B2_B2_B1_A1

### Relational analysis result of NS_B2_A2_A2_B2_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7652359, upper bound: 38.7652359
time: 0.91 seconds

## Relational analysis of NS_B2_A2_A2_B2_A2_B2_B2_B1_A2

### Relational analysis result of NS_B2_A2_A2_B2_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7652359, upper bound: 38.7652359
time: 0.75 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -4.2843742, 15.9625645, -4.0849833, 15.2299109, -19.5142860, 20.0475464
1: -12.0834875, 32.9238968, -11.5078735, 31.4276371, -43.5111237, 44.4317665
2: -18.6558762, 30.3588390, -17.7406940, 29.0492764, -47.7051544, 48.0995331
3: -10.1118202, 38.4398804, -9.6286278, 36.7244186, -46.8362389, 48.0685081
4: -16.7717037, 26.6370869, -15.9518309, 25.4825802, -42.2542839, 42.5889168

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_A2_B2_A2_B2_B2_B2_A1

### Relational analysis result of NS_B2_A2_A2_B2_A2_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7502995, upper bound: 38.7514681
time: 1.16 seconds

## Relational analysis of NS_B2_A2_A2_B2_A2_B2_B2_B2_A2

### Relational analysis result of NS_B2_A2_A2_B2_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7502995, upper bound: 38.7657875
time: 0.97 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 3.15 seconds
NS_B1_A1_A2_B2_A1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.15
Output dim: 4, lower bound: -38.7624641, upper bound: 38.7624640
NS_B1_A1_A2_B2_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.15
Output dim: 4, lower bound: -38.7624641, upper bound: 38.7624640
NS_B1_A1_A2_B2_A1_B1_B2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.15
Output dim: 4, lower bound: -38.7585925, upper bound: 38.7586622
NS_B1_A1_A2_B2_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.15
Output dim: 4, lower bound: -38.7624622, upper bound: 38.7624622
NS_B1_A1_A2_B2_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.15
Output dim: 4, lower bound: -38.7613300, upper bound: 38.7613300
NS_B1_A1_A2_B2_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.15
Output dim: 4, lower bound: -38.7613300, upper bound: 38.7613300
NS_B1_A1_A2_B2_A1_B2_B2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.15
Output dim: 4, lower bound: -38.7486500, upper bound: 38.7516436
NS_B1_A1_A2_B2_A1_B2_B2_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.15
Output dim: 4, lower bound: -38.7486500, upper bound: 38.7486500
NS_B2_A2_A2_B2_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 3.15
Output dim: 4, lower bound: -38.7445396, upper bound: 38.7606986
NS_B2_A2_A2_B2_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 3.15
Output dim: 4, lower bound: -38.7615598, upper bound: 38.7606987
NS_B2_A2_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.15
Output dim: 4, lower bound: -38.7439694, upper bound: 38.7443414
NS_B2_A2_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.15
Output dim: 4, lower bound: -38.7439694, upper bound: 38.7606717
NS_B2_A2_A2_B2_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 3.15
Output dim: 4, lower bound: -38.7620860, upper bound: 38.7618939
NS_B2_A2_A2_B2_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 3.15
Output dim: 4, lower bound: -38.7620860, upper bound: 38.7618939
NS_B2_A2_A2_B2_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.15
Output dim: 4, lower bound: -38.7652359, upper bound: 38.7652359
NS_B2_A2_A2_B2_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.15
Output dim: 4, lower bound: -38.7652359, upper bound: 38.7652359
NS_B2_A2_A2_B2_A2_B2_B2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.15
Output dim: 4, lower bound: -38.7502995, upper bound: 38.7514681
NS_B2_A2_A2_B2_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.15
Output dim: 4, lower bound: -38.7502995, upper bound: 38.7657875

## BFS NS instance: NS_B1_A1_A2_B2_A1_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -3.5341895, 13.1770554, -3.3846812, 12.6328506, -16.1670399, 16.5617352
1: -9.9493599, 27.1768265, -9.5247660, 26.0384464, -35.9878082, 36.7015915
2: -15.3495083, 24.9424057, -14.6906567, 23.9218197, -39.2713280, 39.6330643
3: -8.3230190, 31.7237587, -7.9672227, 30.4239807, -38.7470016, 39.6909790
4: -13.7765541, 21.8856258, -13.1828041, 20.9925900, -34.7691422, 35.0684204

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## BFS NS instance: NS_B1_A1_A2_B2_A1_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -3.5382574, 13.2136326, -3.3846812, 12.6328506, -16.1711082, 16.5983124
1: -9.9822254, 27.2417831, -9.5247660, 26.0384464, -36.0206604, 36.7665482
2: -15.4092531, 25.0002041, -14.6906567, 23.9218197, -39.3310661, 39.6908607
3: -8.3457279, 31.8111267, -7.9672227, 30.4239807, -38.7697067, 39.7783432
4: -13.8289013, 21.9313831, -13.1828041, 20.9925900, -34.8214912, 35.1141739

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## BFS NS instance: NS_B1_A1_A2_B2_A1_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -3.5924911, 13.4181376, -3.3880939, 12.6577854, -16.2502766, 16.8062325
1: -10.1252108, 27.6413002, -9.5504379, 26.0920467, -36.2172585, 37.1917381
2: -15.6069603, 25.4133625, -14.7228165, 24.0534801, -39.6604385, 40.1361771
3: -8.4667149, 32.2803230, -7.9822702, 30.4840355, -38.9507523, 40.2625809
4: -14.0144272, 22.2930279, -13.2119770, 21.0844860, -35.0989075, 35.5050049

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B2_B2_A2_A1

### Relational analysis result of NS_B1_A1_A2_B2_A1_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7624622, upper bound: 38.7624622
time: 1.10 seconds

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B2_B2_A2_A2

### Relational analysis result of NS_B1_A1_A2_B2_A1_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7624622, upper bound: 38.7624621
time: 0.65 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A1_B2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -3.5341895, 13.1770554, -3.3930862, 12.6570673, -16.1912575, 16.5701408
1: -9.9493599, 27.1768265, -9.5597191, 26.1214619, -36.0708237, 36.7365456
2: -15.3495083, 24.9424057, -14.7239027, 23.9441566, -39.2936630, 39.6663055
3: -8.3230190, 31.7237587, -7.9910865, 30.5341930, -38.8572083, 39.7148323
4: -13.7765541, 21.8856258, -13.2074881, 21.0102291, -34.7867813, 35.0931129

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_A1_A2_B2_A1_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_A2_B2_A1_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A2_B2_A1_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_A1_A2_B2_A1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B1_A1_A2_B2_A1_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_A2_B2_A1_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## BFS NS instance: NS_B1_A1_A2_B2_A1_B2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -3.5382574, 13.2136326, -3.3930862, 12.6570673, -16.1953239, 16.6067181
1: -9.9822254, 27.2417831, -9.5597191, 26.1214619, -36.1036758, 36.8015022
2: -15.4092531, 25.0002041, -14.7239027, 23.9441566, -39.3534050, 39.7241058
3: -8.3457279, 31.8111267, -7.9910865, 30.5341930, -38.8799210, 39.8021965
4: -13.8289013, 21.9313831, -13.2074881, 21.0102291, -34.8391304, 35.1388664

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_A1_A2_B2_A1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_A2_B2_A1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B1_A1_A2_B2_A1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A2_B2_A1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B1_A1_A2_B2_A1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_A2_B2_A1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## BFS NS instance: NS_B2_A2_A2_B2_A1_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -3.9940517, 14.9016790, -3.8183870, 14.2483082, -18.2423592, 18.7200661
1: -11.2842083, 30.7247086, -10.7787285, 29.3813343, -40.6655388, 41.5034370
2: -17.4480000, 28.2377701, -16.6356564, 27.1010075, -44.5490074, 44.8734283
3: -9.4433794, 35.8763275, -9.0177612, 34.3496857, -43.7930603, 44.8940849
4: -15.6747637, 24.7712574, -14.9481907, 23.7674141, -39.4421730, 39.7194481

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 32

## BFS NS instance: NS_B2_A2_A2_B2_A1_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -3.9940517, 14.9016790, -3.8222480, 14.2590561, -18.2531071, 18.7239265
1: -11.2842083, 30.7247086, -10.7866459, 29.4137878, -40.6979942, 41.5113525
2: -17.4480000, 28.2377701, -16.6484699, 27.1235657, -44.5715637, 44.8862381
3: -9.4433794, 35.8763275, -9.0243731, 34.3810959, -43.8244743, 44.9006958
4: -15.6747637, 24.7712574, -14.9590473, 23.7915192, -39.4662704, 39.7303047

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 32

## BFS NS instance: NS_B2_A2_A2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.8222480, 14.2590561, -3.8419523, 14.3325024, -18.1547508, 18.1010094
1: -10.7866459, 29.4137878, -10.8419476, 29.5634613, -40.3501015, 40.2557373
2: -16.6484699, 27.1235657, -16.7325363, 27.2673683, -43.9158401, 43.8561020
3: -9.0243731, 34.3810959, -9.0709143, 34.5527687, -43.5771370, 43.4520111
4: -14.9590473, 23.7915192, -15.0358334, 23.9179802, -38.8770294, 38.8273468

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## BFS NS instance: NS_B2_A2_A2_B2_A1_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -3.9940517, 14.9016790, -4.0805969, 15.2210865, -19.2151375, 18.9822769
1: -11.2842083, 30.7247086, -11.5010138, 31.3955746, -42.6797714, 42.2257156
2: -17.4480000, 28.2377701, -17.7326698, 29.0118027, -46.4598007, 45.9704399
3: -9.4433794, 35.8763275, -9.6230412, 36.6963615, -46.1397400, 45.4993668
4: -15.6747637, 24.7712574, -15.9441175, 25.4470749, -41.1218300, 40.7153740

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 32

## BFS NS instance: NS_B2_A2_A2_B2_A1_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -3.9940517, 14.9016790, -4.0872979, 15.2412634, -19.2353153, 18.9889774
1: -11.2842083, 30.7247086, -11.5166426, 31.4484653, -42.7326660, 42.2413521
2: -17.4480000, 28.2377701, -17.7560654, 29.0552979, -46.5032959, 45.9938354
3: -9.4433794, 35.8763275, -9.6358337, 36.7519951, -46.1953659, 45.5121574
4: -15.6747637, 24.7712574, -15.9649792, 25.4873257, -41.1620789, 40.7362366

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 32

## BFS NS instance: NS_B2_A2_A2_B2_A2_B2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -4.3267384, 16.1006126, -4.1591334, 15.4848976, -19.8116322, 20.2597427
1: -12.1824532, 33.2366982, -11.7106962, 31.9519043, -44.1343498, 44.9473953
2: -18.8104057, 30.5870209, -18.0774899, 29.4612045, -48.2716103, 48.6645050
3: -10.1968870, 38.7530823, -9.8021688, 37.2724075, -47.4692955, 48.5552406
4: -16.9035187, 26.8522797, -16.2457333, 25.8559608, -42.7594795, 43.0980148

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_A2_B2_A2_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_A2_A2_B2_A2_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_A2_B2_A2_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B2_A2_A2_B2_A2_B2_B2_B1_A1_A1

### Relational analysis result of NS_B2_A2_A2_B2_A2_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7651914, upper bound: 38.7652005
time: 0.61 seconds

## Relational analysis of NS_B2_A2_A2_B2_A2_B2_B2_B1_A1_A2

### Relational analysis result of NS_B2_A2_A2_B2_A2_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7651914, upper bound: 38.7652073
time: 0.71 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A2_B2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -4.2646980, 15.8843594, -4.1591334, 15.4848976, -19.7495899, 20.0434933
1: -12.0264587, 32.7682953, -11.7106962, 31.9519043, -43.9783554, 44.4789886
2: -18.5682316, 30.2221375, -18.0774899, 29.4612045, -48.0294342, 48.2996216
3: -10.0640373, 38.2615623, -9.8021688, 37.2724075, -47.3364449, 48.0637207
4: -16.6923084, 26.5177250, -16.2457333, 25.8559608, -42.5482712, 42.7634544

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_A2_B2_A2_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_A2_A2_B2_A2_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_A2_A2_B2_A2_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_A2_A2_B2_A2_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B2_A2_A2_B2_A2_B2_B2_B1_A2_A1

### Relational analysis result of NS_B2_A2_A2_B2_A2_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7651914, upper bound: 38.7652005
time: 0.67 seconds

## Relational analysis of NS_B2_A2_A2_B2_A2_B2_B2_B1_A2_A2

### Relational analysis result of NS_B2_A2_A2_B2_A2_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -38.7651914, upper bound: 38.7652073
time: 0.95 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A2_B2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -4.1050639, 15.3071823, -4.0849833, 15.2299109, -19.3349743, 19.3921604
1: -11.5664978, 31.5826912, -11.5078735, 31.4276371, -42.9941330, 43.0905647
2: -17.8315830, 29.1853523, -17.7406940, 29.0492764, -46.8808594, 46.9260406
3: -9.6777325, 36.9075699, -9.6286278, 36.7244186, -46.4021530, 46.5361977
4: -16.0339794, 25.6013775, -15.9518309, 25.4825802, -41.5165596, 41.5532074

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_A2_B2_A2_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B2_A2_A2_B2_A2_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_A2_A2_B2_A2_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B2_A2_A2_B2_A2_B2_B2_B2_A2_B1

### Relational analysis result of NS_B2_A2_A2_B2_A2_B2_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -38.7458841, upper bound: 38.7491712
time: 3.63 seconds

## Relational analysis of NS_B2_A2_A2_B2_A2_B2_B2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_A2_A2_B2_A2_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.62 + 191.39 = 195.01 seconds
