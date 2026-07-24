## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 495.22538199974406


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-226.9094849, 344.3358154, -226.9094849, 344.3358154, -571.2452393, 571.2453003)
1: (-253.4875641, 367.0639648, -253.4875641, 367.0639648, -620.5515137, 620.5515137)
2: (-257.4188232, 361.7965088, -257.4188232, 361.7965088, -619.2153320, 619.2153320)
3: (-309.8564758, 425.3124084, -309.8564758, 425.3124084, -735.1688843, 735.1688843)
4: (-281.1933594, 418.6253052, -281.1933594, 418.6253052, -699.8186646, 699.8186646)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.62 + 2.55 = 3.17 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -495.2650032, upper bound: 495.2650032

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2623076, upper bound: 495.2584724
time: 1.14 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2612235, upper bound: 495.2612235
time: 0.93 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.14 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.14
Output dim: 0, lower bound: -495.2623076, upper bound: 495.2584724
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.14
Output dim: 0, lower bound: -495.2612235, upper bound: 495.2612235

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -220.5705414, 334.5699463, -224.3629608, 340.4119568, -560.9823608, 558.9329224
1: -246.4158478, 356.6816406, -250.6473389, 362.8935547, -609.3092041, 607.3289795
2: -250.2690430, 351.6158447, -254.5464783, 357.7049561, -607.9739990, 606.1622925
3: -301.1424561, 413.3228455, -306.3553772, 420.4967957, -721.6392822, 719.6782227
4: -273.5845642, 406.7001343, -278.1362305, 413.8339844, -687.4185791, 684.8363647

Time for backsubstitution: 0.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2274597, upper bound: 495.2127368
time: 0.87 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.2198811, upper bound: 495.2096994
time: 1.28 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -227.4252167, 346.1140747, -222.6271515, 338.3388977, -565.7640991, 568.7412109
1: -254.1289978, 369.0901184, -248.7605438, 360.6589661, -614.7879639, 617.8506470
2: -258.1418762, 363.7336426, -252.6409607, 355.4639282, -613.6058350, 616.3746338
3: -310.8287048, 427.8486328, -304.1091003, 417.9332886, -728.7619629, 731.9577637
4: -282.9035645, 420.6389160, -276.2780151, 411.1295166, -694.0330811, 696.9169312

Time for backsubstitution: 0.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.2135486, upper bound: 495.2193762
time: 0.88 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.2077494, upper bound: 495.2077494
time: 0.77 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.27 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 2.27
Output dim: 0, lower bound: -495.2274597, upper bound: 495.2127368
NS_A1_A2, status: Status.VERIFIED, split count: 2, time: 2.27
Output dim: 0, lower bound: -495.2198811, upper bound: 495.2096994
NS_A2_B1, status: Status.VERIFIED, split count: 2, time: 2.27
Output dim: 0, lower bound: -495.2135486, upper bound: 495.2193762
NS_A2_B2, status: Status.VERIFIED, split count: 2, time: 2.27
Output dim: 0, lower bound: -495.2077494, upper bound: 495.2077494

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -219.8164368, 333.4123840, -224.3629608, 340.4119568, -560.2281494, 557.7753296
1: -245.5711975, 355.4500427, -250.6473389, 362.8935547, -608.4646606, 606.0973511
2: -249.4164276, 350.3990479, -254.5464783, 357.7049561, -607.1213989, 604.9454346
3: -300.1094360, 411.9019165, -306.3553772, 420.4967957, -720.6062012, 718.2573242
4: -272.6572876, 405.2886353, -278.1362305, 413.8339844, -686.4912720, 683.4248657

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2274597, upper bound: 495.2127368
time: 1.10 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2274597, upper bound: 495.2127368
time: 0.91 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.82 seconds
NS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 0, lower bound: -495.2274597, upper bound: 495.2127368
NS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 0, lower bound: -495.2274597, upper bound: 495.2127368

## BFS NS instance: NS_A1_A1_B1

### Backsubstitution after applying NS history:
0: -219.8164368, 333.4123840, -220.5705414, 334.5699463, -554.3863525, 553.9829102
1: -245.5711975, 355.4500427, -246.4158478, 356.6816406, -602.2528076, 601.8658447
2: -249.4164276, 350.3990479, -250.2690430, 351.6158447, -601.0322876, 600.6679077
3: -300.1094360, 411.9019165, -301.1424561, 413.3228455, -713.4322510, 713.0443726
4: -272.6572876, 405.2886353, -273.5845642, 406.7001343, -679.3574219, 678.8731689

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_A1_B1_B1

### Relational analysis result of NS_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2274597, upper bound: 495.2127368
time: 1.06 seconds

## Relational analysis of NS_A1_A1_B1_B2

### Relational analysis result of NS_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2274597, upper bound: 495.2127368
time: 1.12 seconds

## BFS NS instance: NS_A1_A1_B2

### Backsubstitution after applying NS history:
0: -219.8164368, 333.4123840, -227.4252167, 346.1140747, -565.9303589, 560.8375244
1: -245.5711975, 355.4500427, -254.1289978, 369.0901184, -614.6613159, 609.5790405
2: -249.4164276, 350.3990479, -258.1418762, 363.7336426, -613.1500854, 608.5408325
3: -300.1094360, 411.9019165, -310.8287048, 427.8486328, -727.9580688, 722.7305908
4: -272.6572876, 405.2886353, -282.9035645, 420.6389160, -693.2962036, 688.1921997

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_A1_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2274597, upper bound: 495.2127368
time: 0.95 seconds

## Relational analysis of NS_A1_A1_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2274597, upper bound: 495.2127368
time: 0.97 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.40 seconds
NS_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 0, lower bound: -495.2274597, upper bound: 495.2127368
NS_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 0, lower bound: -495.2274597, upper bound: 495.2127368
NS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 0, lower bound: -495.2274597, upper bound: 495.2127368
NS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.40
Output dim: 0, lower bound: -495.2274597, upper bound: 495.2127368

## BFS NS instance: NS_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -219.8164368, 333.4123840, -219.8164368, 333.4123840, -553.2286987, 553.2286987
1: -245.5711975, 355.4500427, -245.5711975, 355.4500427, -601.0211792, 601.0211792
2: -249.4164276, 350.3990479, -249.4164276, 350.3990479, -599.8154297, 599.8154297
3: -300.1094360, 411.9019165, -300.1094360, 411.9019165, -712.0113525, 712.0113525
4: -272.6572876, 405.2886353, -272.6572876, 405.2886353, -677.9459229, 677.9459229

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_A1_B1_B1_B1

### Relational analysis result of NS_A1_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.2220776, upper bound: 495.2138644
time: 0.91 seconds

## Relational analysis of NS_A1_A1_B1_B1_B2

### Relational analysis result of NS_A1_A1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.2222941, upper bound: 495.2147939
time: 1.06 seconds

## BFS NS instance: NS_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -219.8164368, 333.4123840, -235.7453156, 357.3249207, -577.1411743, 569.1577148
1: -245.5711975, 355.4500427, -263.3038330, 381.1944580, -626.7656250, 618.7538452
2: -249.4164276, 350.3990479, -267.2589111, 375.8688965, -625.2853394, 617.6577759
3: -300.1094360, 411.9019165, -322.0981750, 441.4580688, -741.5675049, 734.0001221
4: -272.6572876, 405.2886353, -291.9901428, 435.2524719, -707.9097290, 697.2788086

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_A1_B1_B2_A1

### Relational analysis result of NS_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2236127, upper bound: 495.2266913
time: 1.01 seconds

## Relational analysis of NS_A1_A1_B1_B2_A2

### Relational analysis result of NS_A1_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.2222941, upper bound: 495.2147939
time: 0.89 seconds

## BFS NS instance: NS_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -219.8164368, 333.4123840, -226.6800690, 344.9667664, -564.7829590, 560.0924072
1: -245.5711975, 355.4500427, -253.2941589, 367.8713379, -613.4425049, 608.7440796
2: -249.4164276, 350.3990479, -257.2996826, 362.5290833, -611.9454956, 607.6986694
3: -300.1094360, 411.9019165, -309.8108521, 426.4418335, -726.5512695, 721.7127686
4: -272.6572876, 405.2886353, -281.9869690, 419.2449951, -691.9022827, 687.2756348

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_A1_B2_B1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2262696, upper bound: 495.2094210
time: 1.14 seconds

## Relational analysis of NS_A1_A1_B2_B1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.2177275, upper bound: 495.2099277
time: 0.82 seconds

## BFS NS instance: NS_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -219.8164368, 333.4123840, -241.9744415, 367.5974426, -587.4137573, 575.3868408
1: -245.5711975, 355.4500427, -270.3169556, 392.2150879, -637.7862549, 625.7669678
2: -249.4164276, 350.3990479, -274.3355408, 386.6886902, -636.1051025, 624.7344360
3: -300.1094360, 411.9019165, -330.9460754, 454.3185730, -754.4279785, 742.8480225
4: -272.6572876, 405.2886353, -300.3384399, 447.8095398, -720.4667358, 705.6270142

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_A1_B2_B2_A1

### Relational analysis result of NS_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2262696, upper bound: 495.2094210
time: 0.98 seconds

## Relational analysis of NS_A1_A1_B2_B2_A2

### Relational analysis result of NS_A1_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.2177275, upper bound: 495.2099277
time: 0.89 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.33 seconds
NS_A1_A1_B1_B1_B1, status: Status.VERIFIED, split count: 5, time: 4.33
Output dim: 0, lower bound: -495.2220776, upper bound: 495.2138644
NS_A1_A1_B1_B1_B2, status: Status.VERIFIED, split count: 5, time: 4.33
Output dim: 0, lower bound: -495.2222941, upper bound: 495.2147939
NS_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -495.2236127, upper bound: 495.2266913
NS_A1_A1_B1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.33
Output dim: 0, lower bound: -495.2222941, upper bound: 495.2147939
NS_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -495.2262696, upper bound: 495.2094210
NS_A1_A1_B2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.33
Output dim: 0, lower bound: -495.2177275, upper bound: 495.2099277
NS_A1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.33
Output dim: 0, lower bound: -495.2262696, upper bound: 495.2094210
NS_A1_A1_B2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.33
Output dim: 0, lower bound: -495.2177275, upper bound: 495.2099277

## BFS NS instance: NS_A1_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -216.9741364, 329.2705994, -235.7453156, 357.3249207, -574.2990723, 565.0159302
1: -242.3949585, 351.0020142, -263.3038330, 381.1944580, -623.5893555, 614.3057861
2: -246.2215576, 346.0379028, -267.2589111, 375.8688965, -622.0904541, 613.2967529
3: -296.2767334, 406.7591248, -322.0981750, 441.4580688, -737.7348022, 728.8572998
4: -269.2077942, 400.1753845, -291.9901428, 435.2524719, -704.4602661, 692.1654053

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_A1_B1_B2_A1_B1

### Relational analysis result of NS_A1_A1_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.2220776, upper bound: 495.2138644
time: 0.88 seconds

## Relational analysis of NS_A1_A1_B1_B2_A1_B2

### Relational analysis result of NS_A1_A1_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.2220776, upper bound: 495.2147939
time: 0.92 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -212.3872223, 321.5995178, -224.3765564, 341.3503723, -553.7374878, 545.9760742
1: -237.3337402, 342.9858704, -250.7406006, 364.0639038, -601.3975830, 593.7264404
2: -241.0557404, 338.1537781, -254.7160645, 358.7788696, -599.8345947, 592.8698120
3: -290.0458679, 397.5389709, -306.7101135, 422.0480347, -712.0938721, 704.2490234
4: -263.8267212, 391.1286316, -279.2637939, 414.9053955, -678.7320557, 670.3923950

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_A1_B2_B1_A1_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2401063, upper bound: 495.2450402
time: 0.83 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2401063, upper bound: 495.2450402
time: 1.12 seconds

## BFS NS instance: NS_A1_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -212.3872223, 321.5995178, -239.5629883, 363.8647156, -576.2519531, 561.1624756
1: -237.3337402, 342.9858704, -267.6399231, 388.2711487, -625.6048584, 610.6257935
2: -241.0557404, 338.1537781, -271.6300659, 382.7684937, -623.8241577, 609.7837524
3: -290.0458679, 397.5389709, -327.6878357, 449.7847900, -739.8305664, 725.2268066
4: -263.8267212, 391.1286316, -297.4850769, 443.2818298, -707.1085205, 688.6136475

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_A1_B2_B2_A1_A1

### Relational analysis result of NS_A1_A1_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.2207483, upper bound: 495.2033571
time: 0.84 seconds

## Relational analysis of NS_A1_A1_B2_B2_A1_A2

### Relational analysis result of NS_A1_A1_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.2158249, upper bound: 495.1983493
time: 0.90 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.97 seconds
NS_A1_A1_B1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.97
Output dim: 0, lower bound: -495.2220776, upper bound: 495.2138644
NS_A1_A1_B1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.97
Output dim: 0, lower bound: -495.2220776, upper bound: 495.2147939
NS_A1_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -495.2401063, upper bound: 495.2450402
NS_A1_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.97
Output dim: 0, lower bound: -495.2401063, upper bound: 495.2450402
NS_A1_A1_B2_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 3.97
Output dim: 0, lower bound: -495.2207483, upper bound: 495.2033571
NS_A1_A1_B2_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 3.97
Output dim: 0, lower bound: -495.2158249, upper bound: 495.1983493

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -212.3872223, 321.5995178, -219.3679810, 333.3662720, -545.7534180, 540.9674072
1: -237.3337402, 342.9858704, -245.1894226, 355.6546021, -592.9883423, 588.1752930
2: -241.0557404, 338.1537781, -249.1009521, 350.5184021, -591.5741577, 587.2546997
3: -290.0458679, 397.5389709, -299.9350281, 412.3504028, -702.3962402, 697.4739990
4: -263.8267212, 391.1286316, -273.3376770, 405.3495483, -669.1762695, 664.4661255

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2557022, upper bound: 495.2417151
time: 1.05 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2603127, upper bound: 495.2509066
time: 0.90 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -212.3872223, 321.5995178, -243.1005249, 369.9232483, -582.3103638, 564.6999512
1: -237.3337402, 342.9858704, -271.7432861, 394.9437256, -632.2774658, 614.7291260
2: -241.0557404, 338.1537781, -276.0828857, 389.0450134, -630.1007690, 614.2366333
3: -290.0458679, 397.5389709, -332.5918579, 458.0405579, -748.0864258, 730.1308594
4: -263.8267212, 391.1286316, -302.7242737, 450.3687134, -714.1954346, 693.8529053

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_B2_B1_A1_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2378620, upper bound: 495.2258100
time: 1.03 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B2_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.2234978, upper bound: 495.1986417
time: 1.00 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 2.68 seconds
NS_A1_A1_B2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 0, lower bound: -495.2557022, upper bound: 495.2417151
NS_A1_A1_B2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 0, lower bound: -495.2603127, upper bound: 495.2509066
NS_A1_A1_B2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.68
Output dim: 0, lower bound: -495.2378620, upper bound: 495.2258100
NS_A1_A1_B2_B1_A1_B2_B2, status: Status.VERIFIED, split count: 7, time: 2.68
Output dim: 0, lower bound: -495.2234978, upper bound: 495.1986417

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -211.0005646, 319.5546570, -217.3949127, 330.1600952, -541.1606445, 536.9495850
1: -235.7891388, 340.8103943, -243.1315002, 352.4027100, -588.1918335, 583.9417725
2: -239.4844666, 336.0012817, -246.8064423, 347.1854248, -586.6699219, 582.8077393
3: -288.1648865, 395.0175476, -297.4582520, 408.5425110, -696.7073975, 692.4758301
4: -262.1389160, 388.6457214, -271.2025757, 401.7078857, -663.8466797, 659.8482056

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2478117, upper bound: 495.2499033
time: 4.78 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2581219, upper bound: 495.2498314
time: 1.06 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -212.3872223, 321.5995178, -218.8721008, 332.6294861, -545.0167236, 540.4716187
1: -237.3337402, 342.9858704, -244.6381378, 354.8681641, -592.2019043, 587.6240234
2: -241.0557404, 338.1537781, -248.5428314, 349.7450867, -590.8007202, 586.6965332
3: -290.0458679, 397.5389709, -299.2619324, 411.4407349, -701.4865723, 696.8009033
4: -263.8267212, 391.1286316, -272.7416077, 404.4457092, -668.2724609, 663.8702393

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2513445, upper bound: 495.2513498
time: 1.04 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2614483, upper bound: 495.2562946
time: 0.98 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -207.0409241, 313.7965393, -227.7312775, 347.3889465, -554.4298706, 541.5278320
1: -231.3078918, 334.5758057, -254.4315491, 370.6651306, -601.9729614, 589.0072632
2: -234.9874878, 329.9239197, -258.6474915, 365.2894287, -600.2769165, 588.5714111
3: -282.6815796, 387.6705322, -311.5834351, 429.6368103, -712.3183594, 699.2539062
4: -257.1814880, 381.5386047, -283.6851501, 422.6592102, -679.8405762, 665.2237549

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_B2_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_A1_B2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A1_B2_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_A1_B2_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A1_B2_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_A1_B2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_B2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_A1_B2_B1_A1_B2_B1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2311178, upper bound: 495.2125269
time: 1.02 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B2_B1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2287475, upper bound: 495.2162739
time: 0.84 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 3.92 seconds
NS_A1_A1_B2_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 0, lower bound: -495.2478117, upper bound: 495.2499033
NS_A1_A1_B2_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 0, lower bound: -495.2581219, upper bound: 495.2498314
NS_A1_A1_B2_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 0, lower bound: -495.2513445, upper bound: 495.2513498
NS_A1_A1_B2_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 0, lower bound: -495.2614483, upper bound: 495.2562946
NS_A1_A1_B2_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 0, lower bound: -495.2311178, upper bound: 495.2125269
NS_A1_A1_B2_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 0, lower bound: -495.2287475, upper bound: 495.2162739

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -204.2245026, 310.3460693, -211.9232178, 322.2379456, -526.4624634, 522.2691650
1: -228.3318176, 331.0385132, -237.0570679, 343.9337463, -572.2655640, 568.0955811
2: -231.7810364, 326.3070679, -240.6113281, 338.8323364, -570.6134033, 566.9183350
3: -279.2470398, 383.5675049, -290.0889282, 398.7153015, -677.9622803, 673.6562500
4: -254.1628876, 377.1661377, -264.5948792, 391.9741211, -646.1368408, 641.7609863

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2478117, upper bound: 495.2482377
time: 1.17 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2478117, upper bound: 495.2498314
time: 1.02 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -209.5284882, 317.4173584, -217.3949127, 330.1600952, -539.6885986, 534.8122559
1: -234.1455078, 338.5121155, -243.1315002, 352.4027100, -586.5482178, 581.6436157
2: -237.8383636, 333.7595825, -246.8064423, 347.1854248, -585.0237427, 580.5660400
3: -286.1809692, 392.3485107, -297.4582520, 408.5425110, -694.7235107, 689.8067627
4: -260.3726501, 386.0309143, -271.2025757, 401.7078857, -662.0805054, 657.2335205

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2580553, upper bound: 495.2482377
time: 1.00 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2580553, upper bound: 495.2498314
time: 1.08 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -205.6274109, 312.4086914, -213.7545471, 325.1774292, -530.8048096, 526.1632080
1: -229.8927917, 333.2325745, -238.9371033, 346.9047852, -576.7976074, 572.1696777
2: -233.3707123, 328.4767456, -242.7155914, 341.8880920, -575.2587891, 571.1923218
3: -281.1436157, 386.1105347, -292.3484497, 402.1937561, -683.3374023, 678.4589844
4: -255.8710022, 379.6669006, -266.5300293, 395.2948303, -651.1658325, 646.1968994

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2513445, upper bound: 495.2484832
time: 1.23 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2513445, upper bound: 495.2513498
time: 1.48 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -210.9105682, 319.4574890, -218.8721008, 332.6294861, -543.5400391, 538.3295898
1: -235.6845856, 340.6823730, -244.6381378, 354.8681641, -590.5527344, 585.3204956
2: -239.4042816, 335.9068298, -248.5428314, 349.7450867, -589.1492310, 584.4495850
3: -288.0550842, 394.8638611, -299.2619324, 411.4407349, -699.4958496, 694.1257935
4: -262.0551453, 388.5064087, -272.7416077, 404.4457092, -666.5008545, 661.2480469

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2580531, upper bound: 495.2484832
time: 1.03 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2580531, upper bound: 495.2562946
time: 1.07 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -205.2174988, 311.0372009, -227.7312775, 347.3889465, -552.6063843, 538.7684326
1: -229.2648315, 331.6103516, -254.4315491, 370.6651306, -599.9299316, 586.0417480
2: -232.9191742, 327.0176086, -258.6474915, 365.2894287, -598.2086182, 585.6651001
3: -280.1887512, 384.2218933, -311.5834351, 429.6368103, -709.8255615, 695.8052979
4: -254.9185486, 378.2086182, -283.6851501, 422.6592102, -677.5777588, 661.8937378

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_B2_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_B2_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_A1_B2_B1_A1_B2_B1_A1_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.1975522, upper bound: 495.1899000
time: 0.86 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B2_B1_A1_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.1975522, upper bound: 495.1878018
time: 1.19 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -227.6560059, 345.0091248, -224.2655792, 342.2282410, -569.8842773, 569.2744751
1: -254.3365173, 368.1319885, -250.5486908, 365.1198120, -619.4562378, 618.6806641
2: -257.9924927, 363.3274841, -254.7317047, 359.8467407, -617.8392334, 618.0592041
3: -311.3965759, 425.6478271, -306.8753662, 423.1854553, -734.5819702, 732.5231323
4: -281.4036255, 420.7904358, -279.4279785, 416.3327942, -697.7363281, 700.2183838

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_B2_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_B2_B1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_A1_B2_B1_A1_B2_B1_A2_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.2085247, upper bound: 495.2032766
time: 0.75 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B2_B1_A2_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.2047288, upper bound: 495.2013828
time: 1.12 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 2.97 seconds
NS_A1_A1_B2_B1_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.97
Output dim: 0, lower bound: -495.2478117, upper bound: 495.2482377
NS_A1_A1_B2_B1_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.97
Output dim: 0, lower bound: -495.2478117, upper bound: 495.2498314
NS_A1_A1_B2_B1_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.97
Output dim: 0, lower bound: -495.2580553, upper bound: 495.2482377
NS_A1_A1_B2_B1_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.97
Output dim: 0, lower bound: -495.2580553, upper bound: 495.2498314
NS_A1_A1_B2_B1_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.97
Output dim: 0, lower bound: -495.2513445, upper bound: 495.2484832
NS_A1_A1_B2_B1_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.97
Output dim: 0, lower bound: -495.2513445, upper bound: 495.2513498
NS_A1_A1_B2_B1_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.97
Output dim: 0, lower bound: -495.2580531, upper bound: 495.2484832
NS_A1_A1_B2_B1_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.97
Output dim: 0, lower bound: -495.2580531, upper bound: 495.2562946
NS_A1_A1_B2_B1_A1_B2_B1_A1_B1, status: Status.VERIFIED, split count: 9, time: 2.97
Output dim: 0, lower bound: -495.1975522, upper bound: 495.1899000
NS_A1_A1_B2_B1_A1_B2_B1_A1_B2, status: Status.VERIFIED, split count: 9, time: 2.97
Output dim: 0, lower bound: -495.1975522, upper bound: 495.1878018
NS_A1_A1_B2_B1_A1_B2_B1_A2_B1, status: Status.VERIFIED, split count: 9, time: 2.97
Output dim: 0, lower bound: -495.2085247, upper bound: 495.2032766
NS_A1_A1_B2_B1_A1_B2_B1_A2_B2, status: Status.VERIFIED, split count: 9, time: 2.97
Output dim: 0, lower bound: -495.2047288, upper bound: 495.2013828

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -204.2245026, 310.3460693, -212.9974060, 324.3595581, -528.5840454, 523.3435059
1: -228.3318176, 331.0385132, -238.3081360, 346.2676392, -574.5994873, 569.3466797
2: -231.7810364, 326.3070679, -241.8199310, 341.1145935, -572.8956299, 568.1266479
3: -279.2470398, 383.5675049, -291.6604004, 401.2675476, -680.5145874, 675.2277222
4: -254.1628876, 377.1661377, -266.0708923, 394.4975891, -648.6604614, 643.2370605

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2463905, upper bound: 495.2473786
time: 0.76 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2434870, upper bound: 495.2313121
time: 0.89 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2474754, upper bound: 495.2480433
time: 0.93 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -204.2245026, 310.3460693, -215.8518829, 327.9013977, -532.1259155, 526.1978760
1: -228.3318176, 331.0385132, -241.4078979, 349.9818115, -578.3135986, 572.4463501
2: -231.7810364, 326.3070679, -245.0806732, 344.8205566, -576.6015625, 571.3876343
3: -279.2470398, 383.5675049, -295.3892212, 405.7305298, -684.9775391, 678.9566040
4: -254.1628876, 377.1661377, -269.3576050, 398.9563293, -653.1192017, 646.5237427

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2463905, upper bound: 495.2481439
time: 0.94 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2478117, upper bound: 495.2497816
time: 0.96 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2478117, upper bound: 495.2499033
time: 0.73 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -209.5284882, 317.4173584, -212.9974060, 324.3595581, -533.8880615, 530.4147339
1: -234.1455078, 338.5121155, -238.3081360, 346.2676392, -580.4131470, 576.8202515
2: -237.8383636, 333.7595825, -241.8199310, 341.1145935, -578.9529419, 575.5794067
3: -286.1809692, 392.3485107, -291.6604004, 401.2675476, -687.4484863, 684.0089111
4: -260.3726501, 386.0309143, -266.0708923, 394.4975891, -654.8702393, 652.1018066

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2535544, upper bound: 495.2469850
time: 1.01 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2578883, upper bound: 495.2464280
time: 0.88 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2573394, upper bound: 495.2463108
time: 0.87 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -209.5284882, 317.4173584, -215.8518829, 327.9013977, -537.4298706, 533.2691040
1: -234.1455078, 338.5121155, -241.4078979, 349.9818115, -584.1273193, 579.9200439
2: -237.8383636, 333.7595825, -245.0806732, 344.8205566, -582.6588745, 578.8402710
3: -286.1809692, 392.3485107, -295.3892212, 405.7305298, -691.9114990, 687.7377319
4: -260.3726501, 386.0309143, -269.3576050, 398.9563293, -659.3289795, 655.3885498

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2578883, upper bound: 495.2481306
time: 0.75 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2573394, upper bound: 495.2484197
time: 0.72 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -205.6274109, 312.4086914, -215.5084076, 328.1593018, -533.7867432, 527.9171143
1: -229.8927917, 333.2325745, -240.9234467, 350.1757202, -580.0684814, 574.1560059
2: -233.3707123, 328.4767456, -244.6734314, 345.0864563, -578.4571533, 573.1501465
3: -281.1436157, 386.1105347, -294.7818298, 405.9582214, -687.1016846, 680.8923340
4: -255.8710022, 379.6669006, -268.7920227, 398.8812256, -654.7521973, 648.4589233

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2496933, upper bound: 495.2475498
time: 0.99 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2492698, upper bound: 495.2451502
time: 1.02 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -205.6274109, 312.4086914, -217.2037354, 330.1935425, -535.8209229, 529.6123657
1: -229.8927917, 333.2325745, -242.7790375, 352.2566223, -582.1494141, 576.0115967
2: -233.3707123, 328.4767456, -246.6857300, 347.1938477, -580.5644531, 575.1624756
3: -281.1436157, 386.1105347, -297.0236511, 408.4255066, -689.5690308, 683.1341553
4: -255.8710022, 379.6669006, -270.7604370, 401.4722595, -657.3432617, 650.4272461

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2492698, upper bound: 495.2437409
time: 0.98 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2492698, upper bound: 495.2451649
time: 0.93 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -210.9105682, 319.4574890, -215.5084076, 328.1593018, -539.0698853, 534.9658813
1: -235.6845856, 340.6823730, -240.9234467, 350.1757202, -585.8602905, 581.6058350
2: -239.4042816, 335.9068298, -244.6734314, 345.0864563, -584.4907227, 580.5802612
3: -288.0550842, 394.8638611, -294.7818298, 405.9582214, -694.0132446, 689.6456909
4: -262.0551453, 388.5064087, -268.7920227, 398.8812256, -660.9363403, 657.2983398

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2545134, upper bound: 495.2475175
time: 1.15 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2550115, upper bound: 495.2468201
time: 1.19 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -210.9105682, 319.4574890, -217.2475433, 330.2554932, -541.1660767, 536.7050171
1: -235.6845856, 340.6823730, -242.8287506, 352.3235779, -588.0081787, 583.5111084
2: -239.4042816, 335.9068298, -246.7336884, 347.2590332, -586.6632080, 582.6405029
3: -288.0550842, 394.8638611, -297.0836487, 408.5021362, -696.5571289, 691.9474487
4: -262.0551453, 388.5064087, -270.8110352, 401.5490112, -663.6041260, 659.3174438

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2304007, upper bound: 495.2550334
time: 1.07 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2564447, upper bound: 495.2424469
time: 1.25 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2428149, upper bound: 495.2436362
time: 1.09 seconds

## Summary of splitting at layer (split count: 9)
- Time for NS candidates: 5.09 seconds
NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 10, time: 5.09
Output dim: 0, lower bound: -495.2434870, upper bound: 495.2313121
NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 10, time: 5.09
Output dim: 0, lower bound: -495.2474754, upper bound: 495.2480433
NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 10, time: 5.09
Output dim: 0, lower bound: -495.2478117, upper bound: 495.2497816
NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 10, time: 5.09
Output dim: 0, lower bound: -495.2478117, upper bound: 495.2499033
NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 10, time: 5.09
Output dim: 0, lower bound: -495.2578883, upper bound: 495.2464280
NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 10, time: 5.09
Output dim: 0, lower bound: -495.2573394, upper bound: 495.2463108
NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 10, time: 5.09
Output dim: 0, lower bound: -495.2578883, upper bound: 495.2481306
NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 10, time: 5.09
Output dim: 0, lower bound: -495.2573394, upper bound: 495.2484197
NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 10, time: 5.09
Output dim: 0, lower bound: -495.2496933, upper bound: 495.2475498
NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 10, time: 5.09
Output dim: 0, lower bound: -495.2492698, upper bound: 495.2451502
NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 10, time: 5.09
Output dim: 0, lower bound: -495.2492698, upper bound: 495.2437409
NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 10, time: 5.09
Output dim: 0, lower bound: -495.2492698, upper bound: 495.2451649
NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 10, time: 5.09
Output dim: 0, lower bound: -495.2545134, upper bound: 495.2475175
NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 10, time: 5.09
Output dim: 0, lower bound: -495.2550115, upper bound: 495.2468201
NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 10, time: 5.09
Output dim: 0, lower bound: -495.2564447, upper bound: 495.2424469
NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 5.09
Output dim: 0, lower bound: -495.2428149, upper bound: 495.2436362

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -202.4295502, 307.6367188, -212.6544037, 323.8437500, -526.2730713, 520.2911377
1: -226.3449097, 328.1492615, -237.9286499, 345.7191467, -572.0640869, 566.0778809
2: -229.7610779, 323.4622803, -241.4347992, 340.5732117, -570.3342285, 564.8969727
3: -276.8184814, 380.2318115, -291.1971741, 400.6343689, -677.4528809, 671.4289551
4: -252.0153198, 373.8649902, -265.6635742, 393.8672180, -645.8825684, 639.5284424

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2428045, upper bound: 495.2312067
time: 0.96 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2428424, upper bound: 495.2304172
time: 0.77 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -202.4967499, 307.7843628, -211.2171478, 321.6709595, -524.1677246, 519.0015259
1: -226.4444580, 328.2985840, -236.3128052, 343.3817749, -569.8262329, 564.6113892
2: -229.9386292, 323.6639709, -239.8033752, 338.2829590, -568.2215576, 563.4672852
3: -276.9543457, 380.4949341, -289.2185669, 397.9256287, -674.8800049, 669.7135010
4: -252.4515991, 373.9444275, -263.8641357, 391.1918030, -643.6433716, 637.8085938

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2509841, upper bound: 495.2480433
time: 0.78 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2509841, upper bound: 495.2480433
time: 1.13 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -202.5571899, 307.6397400, -215.8518829, 327.9013977, -530.4585571, 523.4915771
1: -226.6431580, 328.3113098, -241.4078979, 349.9818115, -576.6250000, 569.7192383
2: -229.9145660, 323.5140991, -245.0806732, 344.8205566, -574.7350464, 568.5947266
3: -277.2511597, 380.3758545, -295.3892212, 405.7305298, -682.9816895, 675.7649536
4: -252.4922333, 374.1273499, -269.3576050, 398.9563293, -651.4484253, 643.4849854

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2477038, upper bound: 495.2469823
time: 0.78 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2451036, upper bound: 495.2493412
time: 1.04 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -205.1343536, 311.6769409, -215.8518829, 327.9013977, -533.0357666, 527.5288086
1: -229.3443146, 332.4508667, -241.4078979, 349.9818115, -579.3261108, 573.8587646
2: -232.8150482, 327.7074585, -245.0806732, 344.8205566, -577.6354980, 572.7881470
3: -280.4739075, 385.2062683, -295.3892212, 405.7305298, -686.2044678, 680.5954590
4: -255.2778015, 378.7691040, -269.3576050, 398.9563293, -654.2341309, 648.1266479

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2440433, upper bound: 495.2498473
time: 1.04 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2451036, upper bound: 495.2494982
time: 0.78 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -206.7353973, 313.3262329, -203.5100250, 310.4521790, -517.1875610, 516.8362427
1: -231.0020142, 334.1160278, -227.6340637, 331.3497009, -562.3516846, 561.7499390
2: -234.6915283, 329.4539490, -231.1669312, 326.4985352, -561.1898193, 560.6208496
3: -282.3339233, 387.2579346, -278.6432800, 384.0953979, -666.4291992, 665.9011841
4: -257.0341492, 380.9723511, -254.7881470, 377.3094482, -634.3436279, 635.7604980

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2569308, upper bound: 495.2378831
time: 1.06 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2561191, upper bound: 495.2460198
time: 1.09 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -205.9980316, 312.2464905, -203.2920380, 309.0418396, -515.0397949, 515.5383911
1: -230.1766205, 332.9455261, -227.3901978, 329.9699097, -560.1464844, 560.3356934
2: -233.9279327, 328.3348694, -230.9496765, 325.2985535, -559.2263794, 559.2845459
3: -281.3291016, 385.9038696, -278.2849121, 382.6977234, -664.0266113, 664.1887817
4: -256.1683960, 379.6332092, -254.2844086, 376.3023071, -632.4707031, 633.9176025

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2565217, upper bound: 495.2377238
time: 0.79 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2559820, upper bound: 495.2459263
time: 1.19 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -206.7353973, 313.3262329, -206.1100616, 313.7576904, -520.4931030, 519.4362793
1: -231.0020142, 334.1160278, -230.4799500, 334.7896729, -565.7916870, 564.5958862
2: -234.6915283, 329.4539490, -234.1985168, 329.9488220, -564.6401978, 563.6524658
3: -282.3339233, 387.2579346, -282.0924683, 388.1448669, -670.4786987, 669.3503418
4: -257.0341492, 380.9723511, -257.8162537, 381.4827271, -638.5168457, 638.7885742

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2570509, upper bound: 495.2395424
time: 1.15 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2559698, upper bound: 495.2476796
time: 0.87 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -205.9980316, 312.2464905, -205.9275360, 312.3114014, -518.3094482, 518.1740112
1: -230.1766205, 332.9455261, -230.2683716, 333.3761902, -563.5527954, 563.2138672
2: -233.9279327, 328.3348694, -233.9405518, 328.6915894, -562.6193848, 562.2753906
3: -281.3291016, 385.9038696, -281.6618652, 386.7279968, -668.0570679, 667.5657349
4: -256.1683960, 379.6332092, -257.3227539, 380.3496704, -636.5180664, 636.9559326

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2567429, upper bound: 495.2398744
time: 0.97 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2559015, upper bound: 495.2480930
time: 0.95 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -201.4664001, 306.1528320, -207.1228790, 315.4900513, -516.9564209, 513.2756958
1: -225.2476501, 326.5253296, -231.5603180, 336.6019592, -561.8496094, 558.0856323
2: -228.6912384, 321.8848572, -235.2543793, 331.7532959, -560.4445190, 557.1392212
3: -275.4480896, 378.4069519, -283.3152161, 390.3967590, -665.8447876, 661.7221680
4: -250.8663635, 371.9098206, -258.7289734, 383.2113037, -634.0776367, 630.6387329

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2309699, upper bound: 495.2329213
time: 1.05 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2491801, upper bound: 495.2471532
time: 1.23 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -200.3987732, 304.6857300, -225.9655762, 345.1451416, -545.5438843, 530.6513062
1: -224.0337677, 324.9439697, -252.4408722, 367.8136292, -591.8472900, 577.3848267
2: -227.5086212, 320.3632812, -256.5989380, 362.6982727, -590.2069092, 576.9620361
3: -273.9709167, 376.5834045, -308.7195740, 426.2989197, -700.2698364, 685.3029175
4: -249.6528015, 370.1623230, -281.2825623, 419.1627502, -668.8153076, 651.4448853

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 25

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2392651, upper bound: 495.2266466
time: 0.81 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2493236, upper bound: 495.2447871
time: 0.92 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -197.5548859, 300.2664490, -212.8383789, 323.6027222, -521.1575928, 513.1046753
1: -220.8805389, 320.2169800, -237.9020844, 345.2015686, -566.0820312, 558.1190796
2: -224.2941132, 315.6910400, -241.7764435, 340.2631836, -564.5572510, 557.4674683
3: -270.0942688, 371.2208252, -291.0501404, 400.3346558, -670.4287720, 662.2708740
4: -246.1731720, 364.6209106, -265.5209351, 393.3121338, -639.4852905, 630.1417847

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2344462, upper bound: 495.2373278
time: 1.24 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2488180, upper bound: 495.2433642
time: 1.09 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -214.7809448, 326.3471069, -212.5391846, 323.2304382, -538.0112305, 538.8862915
1: -239.8836212, 347.7957764, -237.5441437, 344.7962646, -584.6798096, 585.3399048
2: -243.9192963, 343.1553650, -241.4826660, 339.8999634, -583.8192749, 584.6380615
3: -293.1779480, 403.1235352, -290.6483459, 399.9482727, -693.1262207, 693.7718506
4: -266.9936523, 396.3535767, -265.2271729, 392.9305725, -659.9241333, 661.5806274

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2492698, upper bound: 495.2451649
time: 1.23 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2492698, upper bound: 495.2451649
time: 1.22 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -206.6330109, 313.0341187, -207.1228790, 315.4900513, -522.1230469, 520.1569824
1: -230.9162140, 333.8041992, -231.5603180, 336.6019592, -567.5181885, 565.3645020
2: -234.5953369, 329.1463013, -235.2543793, 331.7532959, -566.3485718, 564.4006958
3: -282.2096252, 386.9635620, -283.3152161, 390.3967590, -672.6063843, 670.2788086
4: -256.9259949, 380.5455017, -258.7289734, 383.2113037, -640.1373291, 639.2744751

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2535861, upper bound: 495.2378691
time: 0.97 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2529305, upper bound: 495.2471402
time: 1.03 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -206.1653137, 312.4224243, -225.9655762, 345.1451416, -551.3104248, 538.3880005
1: -230.3743896, 333.1321716, -252.4408722, 367.8136292, -598.1879883, 585.5730591
2: -234.0827942, 328.5173340, -256.5989380, 362.6982727, -596.7810669, 585.1162109
3: -281.5488586, 386.1897278, -308.7195740, 426.2989197, -707.8477783, 694.9093018
4: -256.4090881, 379.8482971, -281.2825623, 419.1627502, -675.5718384, 661.1308594

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2541013, upper bound: 495.2373316
time: 0.98 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2534691, upper bound: 495.2464446
time: 1.00 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -209.9604797, 318.0360718, -217.2475433, 330.2554932, -540.2159424, 535.2836304
1: -234.6314697, 339.1679382, -242.8287506, 352.3235779, -586.9550171, 581.9965210
2: -238.3394623, 334.4129944, -246.7336884, 347.2590332, -585.5984497, 581.1466675
3: -286.7505798, 393.1231079, -297.0836487, 408.5021362, -695.2526855, 690.2066650
4: -260.9258118, 386.7468262, -270.8110352, 401.5490112, -662.4747314, 657.5578613

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2552117, upper bound: 495.2407409
time: 1.18 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2564673, upper bound: 495.2413512
time: 1.07 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -210.0921478, 318.6708069, -215.0178070, 326.8169250, -536.9089355, 533.6885986
1: -234.6600342, 339.8157959, -240.3070984, 348.6544800, -583.3145142, 580.1229248
2: -238.5193329, 334.9943542, -244.1689911, 343.6403198, -582.1596680, 579.1633301
3: -286.8414001, 394.0838928, -293.9656677, 404.3038025, -691.1451416, 688.0495605
4: -261.4909668, 386.9645996, -268.0851746, 397.2769165, -658.7678833, 655.0496216

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2415244, upper bound: 495.2423700
time: 1.13 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2428770, upper bound: 495.2428716
time: 0.86 seconds

## Summary of splitting at layer (split count: 10)
- Time for NS candidates: 2.70 seconds
NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 11, time: 2.70
Output dim: 0, lower bound: -495.2428045, upper bound: 495.2312067
NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 11, time: 2.70
Output dim: 0, lower bound: -495.2428424, upper bound: 495.2304172
NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 11, time: 2.70
Output dim: 0, lower bound: -495.2509841, upper bound: 495.2480433
NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 11, time: 2.70
Output dim: 0, lower bound: -495.2509841, upper bound: 495.2480433
NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 11, time: 2.70
Output dim: 0, lower bound: -495.2477038, upper bound: 495.2469823
NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 11, time: 2.70
Output dim: 0, lower bound: -495.2451036, upper bound: 495.2493412
NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 11, time: 2.70
Output dim: 0, lower bound: -495.2440433, upper bound: 495.2498473
NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 11, time: 2.70
Output dim: 0, lower bound: -495.2451036, upper bound: 495.2494982
NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 11, time: 2.70
Output dim: 0, lower bound: -495.2569308, upper bound: 495.2378831
NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 11, time: 2.70
Output dim: 0, lower bound: -495.2561191, upper bound: 495.2460198
NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 2.70
Output dim: 0, lower bound: -495.2565217, upper bound: 495.2377238
NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 2.70
Output dim: 0, lower bound: -495.2559820, upper bound: 495.2459263
NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 11, time: 2.70
Output dim: 0, lower bound: -495.2570509, upper bound: 495.2395424
NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 2.70
Output dim: 0, lower bound: -495.2559698, upper bound: 495.2476796
NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 2.70
Output dim: 0, lower bound: -495.2567429, upper bound: 495.2398744
NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 2.70
Output dim: 0, lower bound: -495.2559015, upper bound: 495.2480930
NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 11, time: 2.70
Output dim: 0, lower bound: -495.2309699, upper bound: 495.2329213
NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 11, time: 2.70
Output dim: 0, lower bound: -495.2491801, upper bound: 495.2471532
NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 2.70
Output dim: 0, lower bound: -495.2392651, upper bound: 495.2266466
NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 2.70
Output dim: 0, lower bound: -495.2493236, upper bound: 495.2447871
NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 11, time: 2.70
Output dim: 0, lower bound: -495.2344462, upper bound: 495.2373278
NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 11, time: 2.70
Output dim: 0, lower bound: -495.2488180, upper bound: 495.2433642
NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 11, time: 2.70
Output dim: 0, lower bound: -495.2492698, upper bound: 495.2451649
NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 11, time: 2.70
Output dim: 0, lower bound: -495.2492698, upper bound: 495.2451649
NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 11, time: 2.70
Output dim: 0, lower bound: -495.2535861, upper bound: 495.2378691
NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 11, time: 2.70
Output dim: 0, lower bound: -495.2529305, upper bound: 495.2471402
NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 2.70
Output dim: 0, lower bound: -495.2541013, upper bound: 495.2373316
NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 2.70
Output dim: 0, lower bound: -495.2534691, upper bound: 495.2464446
NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 11, time: 2.70
Output dim: 0, lower bound: -495.2552117, upper bound: 495.2407409
NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 11, time: 2.70
Output dim: 0, lower bound: -495.2564673, upper bound: 495.2413512
NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 11, time: 2.70
Output dim: 0, lower bound: -495.2415244, upper bound: 495.2423700
NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 11, time: 2.70
Output dim: 0, lower bound: -495.2428770, upper bound: 495.2428716

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -199.7289124, 303.6049500, -202.1312408, 307.9861450, -507.7150574, 505.7361755
1: -223.3262177, 323.8729858, -226.2011566, 329.0314636, -552.3576660, 550.0740967
2: -226.6802063, 319.2289734, -229.3653564, 323.9508057, -550.6309814, 548.5943604
3: -273.1438904, 375.2594910, -277.0498352, 381.2447205, -654.3885498, 652.3093262
4: -248.6705627, 368.9940491, -252.5253906, 374.7973633, -623.4678345, 621.5194092

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2428045, upper bound: 495.2299849
time: 0.91 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2428045, upper bound: 495.2304172
time: 0.97 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -202.4295502, 307.6367188, -211.8722687, 322.6473999, -525.0769653, 519.5089722
1: -226.3449097, 328.1492615, -237.0585327, 344.4506226, -570.7955322, 565.2077637
2: -229.7610779, 323.4622803, -240.5523834, 339.3188171, -569.0798950, 564.0146484
3: -276.8184814, 380.2318115, -290.1334534, 399.1749878, -675.9934692, 670.3651733
4: -252.0153198, 373.8649902, -264.7290039, 392.3981934, -644.4135132, 638.5938721

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2428424, upper bound: 495.2299849
time: 1.01 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2428424, upper bound: 495.2304172
time: 0.94 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -201.1568146, 305.5744324, -211.2171478, 321.6709595, -522.8276978, 516.7915649
1: -225.1261597, 326.1075745, -236.3128052, 343.3817749, -568.5079346, 562.4204102
2: -228.4422760, 321.3885498, -239.8033752, 338.2829590, -566.7252197, 561.1918945
3: -275.3994446, 377.8771362, -289.2185669, 397.9256287, -673.3250732, 667.0957031
4: -251.1856537, 371.5075073, -263.8641357, 391.1918030, -642.3773804, 635.3716431

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2509808, upper bound: 495.2452689
time: 0.86 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2507634, upper bound: 495.2479328
time: 0.86 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -203.4367218, 309.1537476, -211.2171478, 321.6709595, -525.1076660, 520.3708496
1: -227.4911652, 329.7522888, -236.3128052, 343.3817749, -570.8729248, 566.0650635
2: -231.0063782, 325.1073303, -239.8033752, 338.2829590, -569.2893066, 564.9107056
3: -278.2217712, 382.2104797, -289.2185669, 397.9256287, -676.1472778, 671.4290161
4: -253.6019440, 375.5972900, -263.8641357, 391.1918030, -644.7936401, 639.4614258

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A2_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2493979, upper bound: 495.2479890
time: 1.02 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A2_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2507634, upper bound: 495.2479511
time: 0.80 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -192.1663361, 292.1575623, -213.1678162, 323.8516541, -516.0179443, 505.3253479
1: -215.0917053, 312.0251465, -238.4083557, 345.6939697, -560.7855225, 550.4333496
2: -217.9953308, 307.2680969, -242.0268250, 340.5718079, -558.5670776, 549.2949219
3: -263.3315735, 361.4541321, -291.7288208, 400.7507629, -664.0823364, 653.1829834
4: -239.5794983, 355.5205383, -266.0401917, 394.0475159, -633.6270142, 621.5607300

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1_A1_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2440433, upper bound: 495.2477543
time: 0.99 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2440433, upper bound: 495.2477543
time: 0.91 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -201.7960663, 306.4840088, -215.8518829, 327.9013977, -529.6973877, 522.3358765
1: -225.7955475, 327.0828247, -241.4078979, 349.9818115, -575.7773438, 568.4907227
2: -229.0538940, 322.2999573, -245.0806732, 344.8205566, -573.8743896, 567.3805542
3: -276.2139282, 378.9531555, -295.3892212, 405.7305298, -681.9444580, 674.3423462
4: -251.5787811, 372.7059937, -269.3576050, 398.9563293, -650.5350342, 642.0635376

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1_A2_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2440433, upper bound: 495.2493412
time: 0.84 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1_A2_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2440433, upper bound: 495.2493412
time: 0.79 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -202.4286194, 307.6387329, -206.9401093, 314.5725098, -517.0010986, 514.5788574
1: -226.3199158, 328.1670532, -231.5584259, 336.0415039, -562.3614502, 559.7254028
2: -229.7287903, 323.4689331, -234.8073273, 330.8875732, -560.6162720, 558.2762451
3: -276.7931519, 380.2265930, -283.5458679, 389.4692078, -666.2623291, 663.7724609
4: -251.9279022, 373.8910217, -258.1424561, 382.9737244, -634.9016113, 632.0334473

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2449203, upper bound: 495.2469823
time: 1.16 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2449203, upper bound: 495.2494982
time: 1.14 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -205.1343536, 311.6769409, -215.0372314, 326.6578064, -531.7920532, 526.7141724
1: -229.3443146, 332.4508667, -240.4988098, 348.6601868, -578.0045166, 572.9497070
2: -232.8150482, 327.7074585, -244.1593475, 343.5149841, -576.3300171, 571.8668213
3: -280.4739075, 385.2062683, -294.2758484, 404.2042847, -684.6782227, 679.4820557
4: -255.2778015, 378.7691040, -268.3807068, 397.4260254, -652.7038574, 647.1497803

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2459992, upper bound: 495.2469823
time: 0.81 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2459992, upper bound: 495.2494982
time: 1.04 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -204.9619598, 310.6566467, -203.1694031, 309.9397278, -514.9016724, 513.8259888
1: -229.0370941, 331.2679138, -227.2574921, 330.8038635, -559.8409424, 558.5253906
2: -232.6959991, 326.6544800, -230.7849731, 325.9596863, -558.6557007, 557.4393921
3: -279.9354553, 383.9682617, -278.1833496, 383.4763489, -663.4116211, 662.1515503
4: -254.9075470, 377.7196655, -254.3848877, 376.6824036, -631.5899658, 632.1045532

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2566126, upper bound: 495.2378831
time: 1.01 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2566126, upper bound: 495.2378831
time: 1.13 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -204.9666748, 310.7434082, -201.7263184, 307.7530212, -512.7197266, 512.4697266
1: -229.0998230, 331.3884277, -225.6350250, 328.4506226, -557.5504150, 557.0234375
2: -232.8120728, 326.7565918, -229.1464691, 323.6563110, -556.4681396, 555.9030151
3: -279.9868164, 384.3419800, -276.1967773, 380.7344055, -660.7211914, 660.5385132
4: -255.3683472, 377.6496277, -252.5771179, 373.9939575, -629.3623047, 630.2266846

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2560307, upper bound: 495.2460198
time: 0.92 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2560307, upper bound: 495.2460198
time: 1.08 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -204.2205048, 309.5705261, -202.9523010, 308.5313721, -512.7518311, 512.5228271
1: -228.2071381, 330.0900879, -227.0146332, 329.4257202, -557.6328735, 557.1046753
2: -231.9275970, 325.5277100, -230.5680542, 324.7622375, -556.6898193, 556.0957642
3: -278.9250793, 382.6157532, -277.8246460, 382.0865479, -661.0115967, 660.4402466
4: -254.0364532, 376.3715820, -253.8816071, 375.6789246, -629.7153931, 630.2531738

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2551520, upper bound: 495.2377238
time: 0.95 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2551520, upper bound: 495.2377238
time: 0.80 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -204.2407837, 309.6646423, -201.4997406, 306.3437500, -510.5844727, 511.1642761
1: -228.2773895, 330.2178345, -225.3811493, 327.0676270, -555.3450317, 555.5989990
2: -232.0554504, 325.6388855, -228.9162750, 322.4561768, -554.5114746, 554.5550537
3: -278.9831543, 383.0188904, -275.8207092, 379.3686523, -658.3517456, 658.8395996
4: -254.5098877, 376.3137512, -252.0595703, 372.9877625, -627.4976196, 628.3732910

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2545644, upper bound: 495.2459263
time: 1.33 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2545644, upper bound: 495.2459263
time: 1.02 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -204.9619598, 310.6566467, -205.8163605, 313.3135986, -518.2755737, 516.4729614
1: -229.0370941, 331.2679138, -230.1550293, 334.3169556, -563.3540649, 561.4229736
2: -232.6959991, 326.6544800, -233.8689117, 329.4836426, -562.1796265, 560.5232544
3: -279.9354553, 383.9682617, -281.6967468, 387.6000061, -667.5354614, 665.6650391
4: -254.9075470, 377.7196655, -257.4670105, 380.9430237, -635.8505859, 635.1866455

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2565777, upper bound: 495.2395424
time: 2.42 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2565777, upper bound: 495.2395424
time: 0.99 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -204.9666748, 310.7434082, -203.7133484, 310.1745300, -515.1411743, 514.4567261
1: -229.0998230, 331.3884277, -227.8016205, 330.9435120, -560.0432129, 559.1899414
2: -232.8120728, 326.7565918, -231.4900818, 326.1711426, -558.9830933, 558.2467041
3: -279.9868164, 384.3419800, -278.8022766, 383.7041931, -663.6910400, 663.1441650
4: -255.3683472, 377.6496277, -254.8872986, 377.0457764, -632.4141235, 632.5368652

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2556202, upper bound: 495.2476796
time: 1.14 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2556202, upper bound: 495.2476796
time: 1.03 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -204.2205048, 309.5705261, -205.6316528, 311.8656921, -516.0861816, 515.2021484
1: -228.2071381, 330.0900879, -229.9411621, 332.9013062, -561.1084595, 560.0312500
2: -231.9275970, 325.5277100, -233.6078491, 328.2239685, -560.1515503, 559.1354980
3: -278.9250793, 382.6157532, -281.2624207, 386.1934814, -665.1185303, 663.8779907
4: -254.0364532, 376.3715820, -256.9703369, 379.8075562, -633.8439941, 633.3419189

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2554288, upper bound: 495.2398744
time: 1.04 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2554288, upper bound: 495.2398467
time: 0.90 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -204.2407837, 309.6646423, -203.5089569, 308.7013245, -512.9420776, 513.1735229
1: -228.2773895, 330.2178345, -227.5658722, 329.4960938, -557.7733154, 557.7836914
2: -232.0554504, 325.6388855, -231.2035828, 324.8771973, -556.9325562, 556.8424072
3: -278.9831543, 383.0188904, -278.3286133, 382.3071289, -661.2901611, 661.3474121
4: -254.5098877, 376.3137512, -254.3670959, 375.8774109, -630.3873291, 630.6808472

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2544761, upper bound: 495.2480930
time: 1.08 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2544761, upper bound: 495.2477085
time: 1.33 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B1

### Backsubstitution after applying NS history:
0: -201.1707001, 305.7073059, -205.1396332, 312.5012817, -513.6719360, 510.8468933
1: -224.9204407, 326.0499878, -229.3643799, 333.4201050, -558.3405762, 555.4143677
2: -228.3585815, 321.4174805, -233.0310364, 328.6198425, -556.9783936, 554.4484863
3: -275.0482788, 377.8583679, -280.6290283, 386.7893372, -661.8376465, 658.4874268
4: -250.5127563, 371.3670349, -256.3842163, 379.5499268, -630.0626831, 627.7512207

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.2238946, upper bound: 495.2193128
time: 0.87 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2238946, upper bound: 495.2329213
time: 1.01 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B2

### Backsubstitution after applying NS history:
0: -199.3036652, 302.9444885, -206.4129028, 314.6063538, -513.9099731, 509.3573914
1: -222.8307953, 323.0831909, -230.7986908, 335.6892700, -558.5200806, 553.8818359
2: -226.2520599, 318.4972839, -234.5756683, 330.8614502, -557.1133423, 553.0728149
3: -272.4871216, 374.4461975, -282.4320068, 389.6309509, -662.1180420, 656.8781738
4: -248.2407684, 367.9172363, -258.1958618, 381.9533691, -630.1941528, 626.1129761

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2394386, upper bound: 495.2312175
time: 0.75 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2394386, upper bound: 495.2471532
time: 1.60 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -198.6719055, 302.0718079, -225.6409454, 344.6571655, -543.3290405, 527.7127686
1: -222.1222229, 322.1561890, -252.0812988, 367.2938232, -589.4159546, 574.2374268
2: -225.5649109, 317.6229553, -256.2338257, 362.1868896, -587.7518311, 573.8568115
3: -271.6334534, 373.3660278, -308.2806091, 425.6995544, -697.3329468, 681.6466064
4: -247.5845490, 366.9852600, -280.8960571, 418.5690918, -666.1536255, 647.8813477

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 25

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2392651, upper bound: 495.2265264
time: 1.35 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2392651, upper bound: 495.2266466
time: 0.90 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -198.6857605, 302.1379089, -224.1080780, 342.3481445, -541.0339355, 526.2459106
1: -222.1626129, 322.2167664, -250.3603821, 364.8078003, -586.9703979, 572.5770874
2: -225.6840973, 317.7397766, -254.4929199, 359.7543945, -585.4384766, 572.2326660
3: -271.6908264, 373.7033691, -306.1756592, 422.8213196, -694.5121460, 679.8790283
4: -247.9600677, 366.9488525, -278.9871826, 415.7295532, -663.6895142, 645.9360352

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A2_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2311523, upper bound: 495.2321803
time: 1.04 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A2_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2311523, upper bound: 495.2447871
time: 1.16 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -197.2726135, 299.8406067, -211.0575409, 320.8940125, -518.1666260, 510.8981323
1: -220.5682373, 319.7627869, -235.9269714, 342.3192749, -562.8874512, 555.6896362
2: -223.9766541, 315.2446594, -239.7742157, 337.4260254, -561.4027100, 555.0187378
3: -269.7127380, 370.7002869, -288.6395264, 397.0639648, -666.7767334, 659.3396606
4: -245.8361053, 364.1032410, -263.3986816, 390.0142212, -635.8503418, 627.5019531

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2315898, upper bound: 495.2223181
time: 0.99 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2315898, upper bound: 495.2373279
time: 1.04 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -195.3996124, 297.0694580, -210.7448425, 320.5384827, -515.9381104, 507.8142395
1: -218.4713745, 316.7874146, -235.5841370, 341.9358521, -560.4072266, 552.3715820
2: -221.8628998, 312.3155823, -239.5283203, 337.0643921, -558.9273071, 551.8438721
3: -267.1426086, 367.3073730, -288.2338562, 397.0843201, -664.2269287, 655.5411377
4: -243.5570984, 360.6429138, -263.3347778, 389.4417725, -632.9989014, 623.9776611

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2488180, upper bound: 495.2433642
time: 0.81 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B2_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2488180, upper bound: 495.2433642
time: 1.05 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -214.7809448, 326.3471069, -207.8905792, 316.2138367, -530.9947510, 534.2376709
1: -239.8836212, 347.7957764, -232.3503113, 337.2841187, -577.1676636, 580.1461182
2: -243.9192963, 343.1553650, -236.2242126, 332.4938049, -576.4130859, 579.3795776
3: -293.1779480, 403.1235352, -284.2739868, 391.3093262, -684.4872437, 687.3975220
4: -266.9936523, 396.3535767, -259.6129456, 384.1635132, -651.1569824, 655.9664307

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B1_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2344462, upper bound: 495.2385950
time: 1.45 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B1_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2488180, upper bound: 495.2447791
time: 1.15 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -214.7809448, 326.3471069, -228.1485901, 347.8454285, -562.6263428, 554.4957275
1: -239.8836212, 347.7957764, -254.8331451, 370.6011963, -610.4848022, 602.6289062
2: -243.9192963, 343.1553650, -259.0950623, 365.4994507, -609.4187622, 602.2504272
3: -293.1779480, 403.1235352, -311.5426331, 429.6901245, -722.8677979, 714.6661377
4: -266.9936523, 396.3535767, -283.7773132, 422.4998169, -689.4934692, 680.1307983

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2344462, upper bound: 495.2385950
time: 0.86 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B2_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2488180, upper bound: 495.2447791
time: 0.86 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -204.8689575, 310.3792725, -206.7947388, 314.9959717, -519.8649292, 517.1738892
1: -228.9616547, 330.9710388, -231.1969757, 336.0758362, -565.0374756, 562.1679077
2: -232.6103363, 326.3622131, -234.8864746, 331.2352295, -563.8455811, 561.2485962
3: -279.8240356, 383.7110596, -282.8708496, 389.8002319, -669.6241455, 666.5819092
4: -254.8099823, 377.3103027, -258.3409424, 382.6060486, -637.4160156, 635.6512451

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1_A1_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2342606, upper bound: 495.2260902
time: 0.88 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1_A1_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2342606, upper bound: 495.2378691
time: 1.34 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -204.9004364, 310.5132141, -205.3422394, 312.8103027, -517.7107544, 515.8552856
1: -229.0523224, 331.1402588, -229.5660706, 333.7235107, -562.7758179, 560.7062988
2: -232.7579193, 326.5112610, -233.2379761, 328.9324036, -561.6901245, 559.7490845
3: -279.9011536, 384.1636047, -280.8743591, 387.0595703, -666.9606323, 665.0379639
4: -255.3083344, 377.2882996, -256.5307312, 379.9107666, -635.2190552, 633.8190308

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1_A2_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2342606, upper bound: 495.2331796
time: 0.84 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1_A2_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2342606, upper bound: 495.2471402
time: 1.03 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -204.4179230, 309.7904968, -225.6409454, 344.6571655, -549.0750732, 535.4313965
1: -228.4378052, 330.3235168, -252.0812988, 367.2938232, -595.7316284, 582.4047852
2: -232.1157837, 325.7566223, -256.2338257, 362.1868896, -594.3026123, 581.9904785
3: -279.1851501, 382.9469299, -308.2806091, 425.6995544, -704.8847046, 691.2275391
4: -254.3105774, 376.6421509, -280.8960571, 418.5690918, -672.8796387, 657.5382080

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B2_A1_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2344103, upper bound: 495.2260993
time: 0.94 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B2_A1_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2344103, upper bound: 495.2373316
time: 1.13 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -204.4069672, 309.8435669, -224.1080780, 342.3481445, -546.7551270, 533.9515991
1: -228.4824524, 330.4078674, -250.3603821, 364.8078003, -593.2902222, 580.7682495
2: -232.2142334, 325.8239441, -254.4929199, 359.7543945, -591.9686279, 580.3168335
3: -279.2050781, 383.3006592, -306.1756592, 422.8213196, -702.0263672, 689.4763184
4: -254.7510986, 376.5225525, -278.9871826, 415.7295532, -670.4805298, 655.5097656

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B2_A2_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2344103, upper bound: 495.2331796
time: 0.97 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B2_A2_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2344103, upper bound: 495.2464446
time: 1.01 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -205.6853027, 311.6165771, -208.6221161, 317.2406006, -522.9259033, 520.2387085
1: -229.8660431, 332.2933350, -233.1820984, 338.3900146, -568.2559204, 565.4754639
2: -233.5333405, 327.6558533, -237.0261536, 333.5705872, -567.1039429, 564.6820068
3: -280.9090881, 385.2420654, -285.2673340, 392.5483704, -673.4574585, 670.5093994
4: -255.8003998, 378.7911987, -260.4586182, 385.4276428, -641.2280273, 639.2498169

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2540823, upper bound: 495.2356205
time: 0.92 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2533344, upper bound: 495.2400890
time: 1.31 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -205.3094940, 311.1365051, -228.2455750, 347.9830017, -553.2924805, 539.3820190
1: -229.4258118, 331.7622375, -254.9435730, 370.7498169, -600.1755371, 586.7058105
2: -233.1234131, 327.1647949, -259.2021179, 365.6442871, -598.7677002, 586.3669434
3: -280.3711243, 384.6159058, -311.6756592, 429.8605957, -710.2316284, 696.2915649
4: -255.3912811, 378.2572021, -283.8906250, 422.6695862, -678.0608521, 662.1478271

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 25

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A1_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2548877, upper bound: 495.2376249
time: 1.12 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2260061, upper bound: 495.2252207
time: 1.11 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2546439, upper bound: 495.2398647
time: 0.96 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -205.9756927, 312.5244141, -206.2991028, 313.6852722, -519.6609497, 518.8234863
1: -230.0599365, 333.2287292, -230.5716400, 334.5951538, -564.6550903, 563.8003540
2: -233.8922272, 328.5246582, -234.3725281, 329.8287048, -563.7209473, 562.8971558
3: -281.2206421, 386.6197815, -282.0422058, 388.2567139, -669.4772949, 668.6619263
4: -256.5552673, 379.3418884, -257.6392517, 381.0067139, -637.5619507, 636.9811401

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2_B1_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2413277, upper bound: 495.2423700
time: 1.11 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2_B1_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2413277, upper bound: 495.2423700
time: 1.13 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -205.3184509, 311.5854187, -226.0557861, 344.6254883, -549.9439087, 537.6411743
1: -229.3006744, 332.2088623, -252.4680786, 367.1697083, -596.4703979, 584.6767578
2: -233.1670532, 327.5614624, -256.6978760, 362.1103210, -595.2773438, 584.2593384
3: -280.3081055, 385.4817810, -308.6091003, 425.7311707, -706.0390625, 694.0908813
4: -255.8023834, 378.2322388, -281.2344360, 418.5071716, -674.3095703, 659.4666138

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.0906048, upper bound: 495.0276433
time: 1.03 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.1640483, upper bound: 495.0680229
time: 0.82 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2377112, upper bound: 495.2358581
time: 1.00 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2420241, upper bound: 495.2421827
time: 1.15 seconds

## Summary of splitting at layer (split count: 11)
- Time for NS candidates: 6.12 seconds
NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2428045, upper bound: 495.2299849
NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2428045, upper bound: 495.2304172
NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2428424, upper bound: 495.2299849
NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2428424, upper bound: 495.2304172
NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2509808, upper bound: 495.2452689
NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2507634, upper bound: 495.2479328
NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2493979, upper bound: 495.2479890
NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2507634, upper bound: 495.2479511
NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2440433, upper bound: 495.2477543
NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2440433, upper bound: 495.2477543
NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2440433, upper bound: 495.2493412
NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2440433, upper bound: 495.2493412
NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2449203, upper bound: 495.2469823
NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2449203, upper bound: 495.2494982
NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2459992, upper bound: 495.2469823
NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2459992, upper bound: 495.2494982
NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2566126, upper bound: 495.2378831
NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2566126, upper bound: 495.2378831
NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2560307, upper bound: 495.2460198
NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2560307, upper bound: 495.2460198
NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2551520, upper bound: 495.2377238
NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2551520, upper bound: 495.2377238
NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2545644, upper bound: 495.2459263
NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2545644, upper bound: 495.2459263
NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2565777, upper bound: 495.2395424
NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2565777, upper bound: 495.2395424
NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2556202, upper bound: 495.2476796
NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2556202, upper bound: 495.2476796
NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2554288, upper bound: 495.2398744
NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2554288, upper bound: 495.2398467
NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2544761, upper bound: 495.2480930
NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2544761, upper bound: 495.2477085
NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B1_A1, status: Status.VERIFIED, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2238946, upper bound: 495.2193128
NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2238946, upper bound: 495.2329213
NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2394386, upper bound: 495.2312175
NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2394386, upper bound: 495.2471532
NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2392651, upper bound: 495.2265264
NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2392651, upper bound: 495.2266466
NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2311523, upper bound: 495.2321803
NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2311523, upper bound: 495.2447871
NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2315898, upper bound: 495.2223181
NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2315898, upper bound: 495.2373279
NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2488180, upper bound: 495.2433642
NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2488180, upper bound: 495.2433642
NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2344462, upper bound: 495.2385950
NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2488180, upper bound: 495.2447791
NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2344462, upper bound: 495.2385950
NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2488180, upper bound: 495.2447791
NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2342606, upper bound: 495.2260902
NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2342606, upper bound: 495.2378691
NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2342606, upper bound: 495.2331796
NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2342606, upper bound: 495.2471402
NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2344103, upper bound: 495.2260993
NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2344103, upper bound: 495.2373316
NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2344103, upper bound: 495.2331796
NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2344103, upper bound: 495.2464446
NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2540823, upper bound: 495.2356205
NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2533344, upper bound: 495.2400890
NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2260061, upper bound: 495.2252207
NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2546439, upper bound: 495.2398647
NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2413277, upper bound: 495.2423700
NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2413277, upper bound: 495.2423700
NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2377112, upper bound: 495.2358581
NS_A1_A1_B2_B1_A1_B1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 12, time: 6.12
Output dim: 0, lower bound: -495.2420241, upper bound: 495.2421827

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -190.4755096, 289.9017639, -202.1312408, 307.9861450, -498.4615784, 492.0329895
1: -213.0587311, 309.4211731, -226.2011566, 329.0314636, -542.0902100, 535.6223145
2: -216.0721130, 304.8341675, -229.3653564, 323.9508057, -540.0228271, 534.1995239
3: -260.8456726, 358.4285278, -277.0498352, 381.2447205, -642.0902710, 635.4783936
4: -237.1753387, 352.4967346, -252.5253906, 374.7973633, -611.9725342, 605.0220947

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 42

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -201.6910706, 306.5083008, -202.1312408, 307.9861450, -509.6771851, 508.6395264
1: -225.5221863, 326.9501038, -226.2011566, 329.0314636, -554.5536499, 553.1512451
2: -228.9266205, 322.2774048, -229.3653564, 323.9508057, -552.8774414, 551.6427612
3: -275.8096008, 378.8482361, -277.0498352, 381.2447205, -657.0541992, 655.8980713
4: -251.1299438, 372.4752197, -252.5253906, 374.7973633, -625.9273071, 625.0006104

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 33

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -190.4755096, 289.9017639, -211.8722687, 322.6473999, -513.1229248, 501.7740173
1: -213.0587311, 309.4211731, -237.0585327, 344.4506226, -557.5093384, 546.4797363
2: -216.0721130, 304.8341675, -240.5523834, 339.3188171, -555.3908691, 545.3865356
3: -260.8456726, 358.4285278, -290.1334534, 399.1749878, -660.0206299, 648.5619507
4: -237.1753387, 352.4967346, -264.7290039, 392.3981934, -629.5734863, 617.2257080

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 42

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -201.6910706, 306.5083008, -211.8722687, 322.6473999, -524.3385010, 518.3805542
1: -225.5221863, 326.9501038, -237.0585327, 344.4506226, -569.9727173, 564.0086670
2: -228.9266205, 322.2774048, -240.5523834, 339.3188171, -568.2454224, 562.8297729
3: -275.8096008, 378.8482361, -290.1334534, 399.1749878, -674.9846191, 668.9816895
4: -251.1299438, 372.4752197, -264.7290039, 392.3981934, -643.5281372, 637.2041016

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 38

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A1_A1

### Backsubstitution after applying NS history:
0: -191.1904907, 290.7435913, -208.5072632, 317.5648193, -508.7553101, 499.2508240
1: -214.0791321, 310.5501709, -233.2755585, 339.0321960, -553.1112061, 543.8257446
2: -217.0066833, 305.7900085, -236.7165222, 333.9745789, -550.9811401, 542.5064087
3: -262.0415039, 359.8448181, -285.5091858, 392.8783264, -654.9197998, 645.3540039
4: -238.8471222, 353.6276550, -260.5047913, 386.2142029, -625.0611572, 614.1324463

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A1_A1_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2493979, upper bound: 495.2461613
time: 1.03 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2493979, upper bound: 495.2461613
time: 0.95 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A1_A2

### Backsubstitution after applying NS history:
0: -200.3953705, 304.4169312, -211.2171478, 321.6709595, -522.0662231, 515.6340942
1: -224.2770538, 324.8761292, -236.3128052, 343.3817749, -567.6588135, 561.1889648
2: -227.5814514, 320.1733704, -239.8033752, 338.2829590, -565.8643799, 559.9766846
3: -274.3594055, 376.4555359, -289.2185669, 397.9256287, -672.2849731, 665.6740723
4: -250.2687225, 370.0849609, -263.8641357, 391.1918030, -641.4605103, 633.9490967

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A1_A2_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2493979, upper bound: 495.2479328
time: 1.08 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A1_A2_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2493979, upper bound: 495.2479328
time: 0.79 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -200.7480469, 305.1473083, -200.7166901, 305.8415527, -506.5895996, 505.8639526
1: -224.4840851, 325.5021667, -224.6110229, 326.7239075, -551.2078857, 550.1131592
2: -227.9380951, 320.8966370, -227.7601776, 321.6895142, -549.6275024, 548.6567993
3: -274.5600891, 377.2540283, -275.0989990, 378.5765686, -653.1366577, 652.3528442
4: -250.2714539, 370.7455750, -250.7580719, 372.1549683, -622.4263916, 621.5036011

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A2_B1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2501142, upper bound: 495.2452689
time: 1.03 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A2_B1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2501142, upper bound: 495.2479511
time: 0.93 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -203.4367218, 309.1537476, -210.4214020, 320.4526672, -523.8894043, 519.5751343
1: -227.4911652, 329.7522888, -235.4277954, 342.0895081, -569.5806274, 565.1800537
2: -231.0063782, 325.1073303, -238.9052277, 337.0059814, -568.0123291, 564.0125732
3: -278.2217712, 382.2104797, -288.1354980, 396.4317932, -674.6535034, 670.3459473
4: -253.6019440, 375.5972900, -262.9129639, 389.6968689, -643.2985229, 638.5102539

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A2_B2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2509694, upper bound: 495.2452689
time: 0.84 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A2_B2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A1_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2509694, upper bound: 495.2479511
time: 1.03 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -192.1663361, 292.1575623, -206.9401093, 314.5725098, -506.7388000, 499.0976257
1: -215.0917053, 312.0251465, -231.5584259, 336.0415039, -551.1331787, 543.5834961
2: -217.9953308, 307.2680969, -234.8073273, 330.8875732, -548.8827515, 542.0754395
3: -263.3315735, 361.4541321, -283.5458679, 389.4692078, -652.8007812, 645.0000000
4: -239.5794983, 355.5205383, -258.1424561, 382.9737244, -622.5532227, 613.6629639

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -192.1663361, 292.1575623, -215.0372314, 326.6578064, -518.8240356, 507.1947021
1: -215.0917053, 312.0251465, -240.4988098, 348.6601868, -563.7518311, 552.5238647
2: -217.9953308, 307.2680969, -244.1593475, 343.5149841, -561.5103149, 551.4274292
3: -263.3315735, 361.4541321, -294.2758484, 404.2042847, -667.5358276, 655.7299805
4: -239.5794983, 355.5205383, -268.3807068, 397.4260254, -637.0054321, 623.9012451

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 49

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -201.7960663, 306.4840088, -206.9401093, 314.5725098, -516.3684692, 513.4241333
1: -225.7955475, 327.0828247, -231.5584259, 336.0415039, -561.8370361, 558.6412354
2: -229.0538940, 322.2999573, -234.8073273, 330.8875732, -559.9414062, 557.1072388
3: -276.2139282, 378.9531555, -283.5458679, 389.4692078, -665.6831055, 662.4990234
4: -251.5787811, 372.7059937, -258.1424561, 382.9737244, -634.5524902, 630.8484497

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 39

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -201.7960663, 306.4840088, -215.0372314, 326.6578064, -528.4536743, 521.5212402
1: -225.7955475, 327.0828247, -240.4988098, 348.6601868, -574.4557495, 567.5816650
2: -229.0538940, 322.2999573, -244.1593475, 343.5149841, -572.5688477, 566.4592285
3: -276.2139282, 378.9531555, -294.2758484, 404.2042847, -680.4182129, 673.2288818
4: -251.5787811, 372.7059937, -268.3807068, 397.4260254, -649.0046997, 641.0866699

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 39

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -193.0555267, 293.7626343, -206.9401093, 314.5725098, -507.6280518, 500.7026978
1: -215.9168701, 313.5343018, -231.5584259, 336.0415039, -551.9583740, 545.0927124
2: -218.9848633, 308.8916626, -234.8073273, 330.8875732, -549.8723755, 543.6989746
3: -264.3269958, 363.1821289, -283.5458679, 389.4692078, -653.7962036, 646.7279663
4: -240.2915039, 357.1721802, -258.1424561, 382.9737244, -623.2652588, 615.3146362

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 42

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -204.3963776, 310.5494080, -206.9401093, 314.5725098, -518.9688721, 517.4895020
1: -228.5220337, 331.2530212, -231.5584259, 336.0415039, -564.5635376, 562.8114624
2: -231.9812164, 326.5234070, -234.8073273, 330.8875732, -562.8686523, 561.3307495
3: -279.4654236, 383.8233948, -283.5458679, 389.4692078, -668.9346313, 667.3692627
4: -254.3928833, 377.3806152, -258.1424561, 382.9737244, -637.3665771, 635.5230103

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 39

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -193.0555267, 293.7626343, -215.0372314, 326.6578064, -519.7132568, 508.7997742
1: -215.9168701, 313.5343018, -240.4988098, 348.6601868, -564.5770264, 554.0330811
2: -218.9848633, 308.8916626, -244.1593475, 343.5149841, -562.4998779, 553.0510254
3: -264.3269958, 363.1821289, -294.2758484, 404.2042847, -668.5312500, 657.4578247
4: -240.2915039, 357.1721802, -268.3807068, 397.4260254, -637.7174072, 625.5528564

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 27

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -204.3963776, 310.5494080, -215.0372314, 326.6578064, -531.0540771, 525.5866699
1: -228.5220337, 331.2530212, -240.4988098, 348.6601868, -577.1822510, 571.7518311
2: -231.9812164, 326.5234070, -244.1593475, 343.5149841, -575.4962158, 570.6827393
3: -279.4654236, 383.8233948, -294.2758484, 404.2042847, -683.6696777, 678.0991211
4: -254.3928833, 377.3806152, -268.3807068, 397.4260254, -651.8187866, 645.7612915

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 39

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -198.2842560, 300.8479919, -203.1694031, 309.9397278, -508.2239990, 504.0173340
1: -221.5271759, 320.7344360, -227.2574921, 330.8038635, -552.3310547, 547.9918213
2: -225.1753845, 316.3388672, -230.7849731, 325.9596863, -551.1350708, 547.1238403
3: -270.7384033, 371.8997192, -278.1833496, 383.4763489, -654.2146606, 650.0829468
4: -246.9326935, 365.6181946, -254.3848877, 376.6824036, -623.6151123, 620.0030518

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A1_A1_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2563577, upper bound: 495.2377711
time: 0.87 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2566126, upper bound: 495.2372479
time: 0.92 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -198.1522675, 300.0877991, -203.1694031, 309.9397278, -508.0919495, 503.2571716
1: -221.3369904, 320.0122986, -227.2574921, 330.8038635, -552.1408081, 547.2697754
2: -225.1061859, 315.6969299, -230.7849731, 325.9596863, -551.0658569, 546.4819336
3: -270.5072327, 371.2229919, -278.1833496, 383.4763489, -653.9835815, 649.4063721
4: -246.6844177, 365.1283569, -254.3848877, 376.6824036, -623.3667603, 619.5132446

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 21

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A1_A2_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2563577, upper bound: 495.2377711
time: 1.02 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A1_A2_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2566126, upper bound: 495.2372479
time: 1.37 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -198.4040222, 301.1816406, -201.7263184, 307.7530212, -506.1570435, 502.9079285
1: -221.7213440, 321.1053467, -225.6350250, 328.4506226, -550.1719360, 546.7403564
2: -225.4258881, 316.6876221, -229.1464691, 323.6563110, -549.0821533, 545.8339844
3: -270.9606934, 372.6015930, -276.1967773, 380.7344055, -651.6950684, 648.7981567
4: -247.5381165, 365.8334351, -252.5771179, 373.9939575, -621.5320435, 618.4105225

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 21

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A2_A1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2532957, upper bound: 495.2433223
time: 0.82 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A2_A1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2532957, upper bound: 495.2460197
time: 0.87 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -198.2181244, 300.4183655, -201.7263184, 307.7530212, -505.9711304, 502.1446228
1: -221.4669342, 320.3069763, -225.6350250, 328.4506226, -549.9175415, 545.9419556
2: -225.2881470, 315.9844055, -229.1464691, 323.6563110, -548.9442749, 545.1307983
3: -270.6290588, 371.9436951, -276.1967773, 380.7344055, -651.3634644, 648.1403198
4: -247.2173920, 365.3004150, -252.5771179, 373.9939575, -621.2112427, 617.8775635

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A2_A2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2532957, upper bound: 495.2433223
time: 0.88 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A2_A2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2532957, upper bound: 495.2460198
time: 1.00 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -198.2842560, 300.8479919, -202.9523010, 308.5313721, -506.8156128, 503.8002930
1: -221.5271759, 320.7344360, -227.0146332, 329.4257202, -550.9528809, 547.7488403
2: -225.1753845, 316.3388672, -230.5680542, 324.7622375, -549.9376221, 546.9069214
3: -270.7384033, 371.8997192, -277.8246460, 382.0865479, -652.8249512, 649.7243652
4: -246.9326935, 365.6181946, -253.8816071, 375.6789246, -622.6116333, 619.4998169

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_A1_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2551520, upper bound: 495.2376192
time: 1.20 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 38

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -198.1522675, 300.0877991, -202.9523010, 308.5313721, -506.6836548, 503.0401001
1: -221.3369904, 320.0122986, -227.0146332, 329.4257202, -550.7626953, 547.0268555
2: -225.1061859, 315.6969299, -230.5680542, 324.7622375, -549.8684082, 546.2650146
3: -270.5072327, 371.2229919, -277.8246460, 382.0865479, -652.5937500, 649.0476074
4: -246.6844177, 365.1283569, -253.8816071, 375.6789246, -622.3632202, 619.0099487

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_A2_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2551520, upper bound: 495.2376192
time: 1.12 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 42

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -198.4040222, 301.1816406, -201.4997406, 306.3437500, -504.7477722, 502.6813660
1: -221.7213440, 321.1053467, -225.3811493, 327.0676270, -548.7889404, 546.4865112
2: -225.4258881, 316.6876221, -228.9162750, 322.4561768, -547.8819580, 545.6038208
3: -270.9606934, 372.6015930, -275.8207092, 379.3686523, -650.3293457, 648.4223022
4: -247.5381165, 365.8334351, -252.0595703, 372.9877625, -620.5258789, 617.8930054

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A2_A1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2523692, upper bound: 495.2432923
time: 1.05 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A2_A1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2523692, upper bound: 495.2459263
time: 1.17 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -198.2181244, 300.4183655, -201.4997406, 306.3437500, -504.5618591, 501.9180908
1: -221.4669342, 320.3069763, -225.3811493, 327.0676270, -548.5345459, 545.6881104
2: -225.2881470, 315.9844055, -228.9162750, 322.4561768, -547.7441406, 544.9005737
3: -270.6290588, 371.9436951, -275.8207092, 379.3686523, -649.9976807, 647.7644043
4: -247.2173920, 365.3004150, -252.0595703, 372.9877625, -620.2050781, 617.3599854

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A2_A2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2523692, upper bound: 495.2432923
time: 0.81 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A2_A2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2523692, upper bound: 495.2459263
time: 1.00 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -198.2842560, 300.8479919, -205.8163605, 313.3135986, -511.5978394, 506.6642761
1: -221.5271759, 320.7344360, -230.1550293, 334.3169556, -555.8441162, 550.8893433
2: -225.1753845, 316.3388672, -233.8689117, 329.4836426, -554.6590576, 550.2077637
3: -270.7384033, 371.8997192, -281.6967468, 387.6000061, -658.3383789, 653.5964355
4: -246.9326935, 365.6181946, -257.4670105, 380.9430237, -627.8757324, 623.0852051

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A1_A1_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2562744, upper bound: 495.2394979
time: 0.96 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2565777, upper bound: 495.2385883
time: 1.01 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -198.1522675, 300.0877991, -205.8163605, 313.3135986, -511.4658813, 505.9041138
1: -221.3369904, 320.0122986, -230.1550293, 334.3169556, -555.6539307, 550.1672974
2: -225.1061859, 315.6969299, -233.8689117, 329.4836426, -554.5898438, 549.5657959
3: -270.5072327, 371.2229919, -281.6967468, 387.6000061, -658.1072388, 652.9197388
4: -246.6844177, 365.1283569, -257.4670105, 380.9430237, -627.6273804, 622.5953369

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A1_A2_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2562744, upper bound: 495.2394979
time: 0.78 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A1_A2_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2565777, upper bound: 495.2385883
time: 1.10 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -198.4040222, 301.1816406, -203.7133484, 310.1745300, -508.5785522, 504.8949890
1: -221.7213440, 321.1053467, -227.8016205, 330.9435120, -552.6647339, 548.9069214
2: -225.4258881, 316.6876221, -231.4900818, 326.1711426, -551.5970459, 548.1777344
3: -270.9606934, 372.6015930, -278.8022766, 383.7041931, -654.6649170, 651.4038086
4: -247.5381165, 365.8334351, -254.8872986, 377.0457764, -624.5838623, 620.7207031

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A2_A1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2531507, upper bound: 495.2444745
time: 1.06 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A2_A1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2531507, upper bound: 495.2476796
time: 1.26 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -198.2181244, 300.4183655, -203.7133484, 310.1745300, -508.3926086, 504.1316833
1: -221.4669342, 320.3069763, -227.8016205, 330.9435120, -552.4104004, 548.1084595
2: -225.2881470, 315.9844055, -231.4900818, 326.1711426, -551.4592285, 547.4744873
3: -270.6290588, 371.9436951, -278.8022766, 383.7041931, -654.3332520, 650.7459717
4: -247.2173920, 365.3004150, -254.8872986, 377.0457764, -624.2631836, 620.1877441

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A2_A2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2531507, upper bound: 495.2444745
time: 0.99 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A2_A2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2531507, upper bound: 495.2476796
time: 1.09 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -198.2842560, 300.8479919, -205.6316528, 311.8656921, -510.1499634, 506.4796448
1: -221.5271759, 320.7344360, -229.9411621, 332.9013062, -554.4284668, 550.6755981
2: -225.1753845, 316.3388672, -233.6078491, 328.2239685, -553.3993530, 549.9467163
3: -270.7384033, 371.8997192, -281.2624207, 386.1934814, -656.9318848, 653.1621094
4: -246.9326935, 365.6181946, -256.9703369, 379.8075562, -626.7402344, 622.5885010

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A1_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2554288, upper bound: 495.2398287
time: 0.87 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 38

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -198.1522675, 300.0877991, -205.6316528, 311.8656921, -510.0179443, 505.7194519
1: -221.3369904, 320.0122986, -229.9411621, 332.9013062, -554.2382812, 549.9534912
2: -225.1061859, 315.6969299, -233.6078491, 328.2239685, -553.3301392, 549.3047485
3: -270.5072327, 371.2229919, -281.2624207, 386.1934814, -656.7006836, 652.4854126
4: -246.6844177, 365.1283569, -256.9703369, 379.8075562, -626.4917603, 622.0986938

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A2_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2554288, upper bound: 495.2398005
time: 0.93 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -198.4040222, 301.1816406, -203.5089569, 308.7013245, -507.1053467, 504.6906128
1: -221.7213440, 321.1053467, -227.5658722, 329.4960938, -551.2172852, 548.6712036
2: -225.4258881, 316.6876221, -231.2035828, 324.8771973, -550.3030396, 547.8911133
3: -270.9606934, 372.6015930, -278.3286133, 382.3071289, -653.2678223, 650.9300537
4: -247.5381165, 365.8334351, -254.3670959, 375.8774109, -623.4155273, 620.2004395

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A2_A1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2523692, upper bound: 495.2449419
time: 0.76 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A2_A1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2523692, upper bound: 495.2480930
time: 0.95 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -198.2181244, 300.4183655, -203.5089569, 308.7013245, -506.9194031, 503.9273071
1: -221.4669342, 320.3069763, -227.5658722, 329.4960938, -550.9629517, 547.8728638
2: -225.2881470, 315.9844055, -231.2035828, 324.8771973, -550.1652222, 547.1879272
3: -270.6290588, 371.9436951, -278.3286133, 382.3071289, -652.9361572, 650.2722778
4: -247.2173920, 365.3004150, -254.3670959, 375.8774109, -623.0947876, 619.6674805

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A2_A2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2523692, upper bound: 495.2444745
time: 0.88 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A2_A2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B1_A2_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2523692, upper bound: 495.2477085
time: 0.91 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -199.8028107, 303.6927490, -205.1396332, 312.5012817, -512.3040161, 508.8323364
1: -223.4327850, 323.8951416, -229.3643799, 333.4201050, -556.8527832, 553.2593384
2: -226.9223938, 319.3525391, -233.0310364, 328.6198425, -555.5422363, 552.3835449
3: -273.2431335, 375.5787354, -280.6290283, 386.7893372, -660.0324707, 656.2077637
4: -249.2384796, 368.8127747, -256.3842163, 379.5499268, -628.7883911, 625.1970215

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B1_A2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2238946, upper bound: 495.2323156
time: 1.06 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B1_A2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2238946, upper bound: 495.2329213
time: 1.03 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -199.6736908, 303.4496765, -206.4129028, 314.6063538, -514.2799683, 509.8625793
1: -223.2635345, 323.6417542, -230.7986908, 335.6892700, -558.9528198, 554.4404297
2: -226.6743011, 319.0497437, -234.5756683, 330.8614502, -557.5355835, 553.6253052
3: -273.0240479, 375.0794373, -282.4320068, 389.6309509, -662.6550293, 657.5113525
4: -248.7226410, 368.6177063, -258.1958618, 381.9533691, -630.6760254, 626.8134766

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B2_A1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2238946, upper bound: 495.2311953
time: 1.06 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B2_A1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2238946, upper bound: 495.2312175
time: 1.23 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -199.8028107, 303.6927490, -206.4129028, 314.6063538, -514.4091187, 510.1056519
1: -223.4327850, 323.8951416, -230.7986908, 335.6892700, -559.1220093, 554.6936646
2: -226.9223938, 319.3525391, -234.5756683, 330.8614502, -557.7837524, 553.9281616
3: -273.2431335, 375.5787354, -282.4320068, 389.6309509, -662.8740845, 658.0107422
4: -249.2384796, 368.8127747, -258.1958618, 381.9533691, -631.1918335, 627.0086060

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B2_A2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2238946, upper bound: 495.2469022
time: 0.97 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B2_A2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2238946, upper bound: 495.2471532
time: 0.98 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -194.8511963, 296.2972717, -225.6409454, 344.6571655, -539.5083008, 521.9382324
1: -217.8617706, 315.9682922, -252.0812988, 367.2938232, -585.1555176, 568.0495605
2: -221.2618561, 311.5207520, -256.2338257, 362.1868896, -583.4486694, 567.7545166
3: -266.4111938, 366.3416138, -308.2806091, 425.6995544, -692.1107178, 674.6221924
4: -242.9623566, 359.7572327, -280.8960571, 418.5690918, -661.5314331, 640.6533203

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 27

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -213.1355438, 323.8664551, -225.6409454, 344.6571655, -557.7927246, 549.5073853
1: -238.0610199, 345.1512756, -252.0812988, 367.2938232, -605.3547974, 597.2325439
2: -242.0627747, 340.5549927, -256.2338257, 362.1868896, -604.2496338, 596.7886963
3: -290.9523010, 400.0694275, -308.2806091, 425.6995544, -716.6518555, 708.3500366
4: -265.0195007, 393.3532104, -280.8960571, 418.5690918, -683.5886230, 674.2492676

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 39

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -198.6857605, 302.1379089, -223.9971313, 342.1886597, -540.8743896, 526.1350098
1: -222.1626129, 322.2167664, -250.2602539, 364.6647034, -586.8273315, 572.4769897
2: -225.6840973, 317.7397766, -254.3853455, 359.5994263, -585.2835083, 572.1251221
3: -271.6908264, 373.7033691, -306.0581970, 422.6679077, -694.3587646, 679.7615967
4: -247.9600677, 366.9488525, -278.9406128, 415.5640564, -663.5241089, 645.8894043

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A2_B1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2309699, upper bound: 495.2310379
time: 0.91 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A2_B1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2309699, upper bound: 495.2321803
time: 0.92 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -198.6857605, 302.1379089, -225.0771179, 343.9140015, -542.5997314, 527.2150269
1: -222.1626129, 322.2167664, -251.4845734, 366.4794312, -588.6420288, 573.7012329
2: -225.6840973, 317.7397766, -255.6950378, 361.4847412, -587.1687622, 573.4346924
3: -271.6908264, 373.7033691, -307.6209717, 424.7913513, -696.4821777, 681.3242798
4: -247.9600677, 366.9488525, -280.5097656, 417.7117920, -665.6718750, 647.4585571

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A2_B2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2309699, upper bound: 495.2433733
time: 1.17 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A2_B2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2309699, upper bound: 495.2447871
time: 1.02 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -195.8434906, 297.6814880, -211.0575409, 320.8940125, -516.7374878, 508.7390137
1: -218.9868011, 317.4617004, -235.9269714, 342.3192749, -561.3060303, 553.3886719
2: -222.3689270, 312.9816589, -239.7742157, 337.4260254, -559.7949219, 552.7557373
3: -267.7810364, 368.0616150, -288.6395264, 397.0639648, -664.8448486, 656.7009888
4: -244.1283112, 361.4789124, -263.3986816, 390.0142212, -634.1424561, 624.8775635

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2315898, upper bound: 495.2223181
time: 0.84 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2315898, upper bound: 495.2223181
time: 0.94 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -195.9703674, 297.9865723, -211.0575409, 320.8940125, -516.8643799, 509.0440979
1: -219.1532593, 317.7702942, -235.9269714, 342.3192749, -561.4724731, 553.6972656
2: -222.6162109, 313.3325806, -239.7742157, 337.4260254, -560.0422363, 553.1066895
3: -267.9960938, 368.6287231, -288.6395264, 397.0639648, -665.0600586, 657.2680054
4: -244.6461945, 361.7225647, -263.3986816, 390.0142212, -634.6604004, 625.1212158

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2315898, upper bound: 495.2373278
time: 0.87 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2315898, upper bound: 495.2373278
time: 1.03 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -195.3996124, 297.0694580, -206.5110016, 314.2519836, -509.6515808, 503.5804443
1: -218.4713745, 316.7874146, -230.8359222, 335.1877441, -553.6591187, 547.6233521
2: -221.8628998, 312.3155823, -234.7569122, 330.4094238, -552.2723389, 547.0724487
3: -267.1426086, 367.3073730, -282.4208069, 389.4103394, -656.5529785, 649.7280273
4: -243.5570984, 360.6429138, -258.2545166, 381.5911865, -625.1483154, 618.8974609

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B2_B1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2315898, upper bound: 495.2264958
time: 0.96 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B2_B1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2315898, upper bound: 495.2433642
time: 1.04 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -195.3996124, 297.0694580, -225.7659760, 344.3264771, -539.7260742, 522.8353882
1: -218.4713745, 316.7874146, -252.2003479, 366.8069153, -585.2782593, 568.9877319
2: -221.8628998, 312.3155823, -256.4898987, 361.8593445, -583.7222290, 568.8054199
3: -267.1426086, 367.3073730, -308.3872681, 425.4865417, -692.6291504, 675.6945190
4: -243.5570984, 360.6429138, -281.1808777, 418.2401428, -661.7972412, 641.8237915

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B2_B2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2315898, upper bound: 495.2264958
time: 0.98 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B2_B2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2315898, upper bound: 495.2433642
time: 0.96 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -214.5039062, 325.9298706, -206.1700592, 313.5954895, -528.0993652, 532.0999146
1: -239.5767975, 347.3508911, -230.4382172, 334.4965820, -574.0733032, 577.7891235
2: -243.6068115, 342.7178040, -234.2865906, 329.7528992, -573.3596191, 577.0042725
3: -292.8030701, 402.6095581, -281.9414062, 388.1670227, -680.9700317, 684.5509644
4: -266.6614990, 395.8483582, -257.5604553, 380.9779053, -647.6392822, 653.4087524

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B1_B1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2238946, upper bound: 495.2265243
time: 0.87 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B1_B1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2283458, upper bound: 495.2432467
time: 1.11 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -212.5829620, 323.0835876, -205.8054810, 313.2803040, -525.8632812, 528.8890381
1: -237.4247437, 344.2901917, -230.0175018, 334.1385193, -571.5631714, 574.3076782
2: -241.4355774, 339.7105408, -233.9685669, 329.3862305, -570.8216553, 573.6790771
3: -290.1712646, 399.1027527, -281.4435120, 388.2048340, -678.3760986, 680.5462646
4: -264.3122253, 392.3148499, -257.4229431, 380.3542175, -644.6664429, 649.7377930

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B1_B2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2396946, upper bound: 495.2315424
time: 1.03 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B1_B2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2396946, upper bound: 495.2497207
time: 0.90 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -214.5039062, 325.9298706, -226.4535522, 345.2748108, -559.7786865, 552.3833008
1: -239.5767975, 347.3508911, -252.9554443, 367.8675537, -607.4442749, 600.3062744
2: -243.6068115, 342.7178040, -257.1857605, 362.8113098, -606.4180908, 599.9035645
3: -292.8030701, 402.6095581, -309.2547607, 426.5385132, -719.3415527, 711.8642578
4: -266.6614990, 395.8483582, -281.7534790, 419.3924866, -686.0539551, 677.6017456

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B2_B1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.2238946, upper bound: 495.2226644
time: 1.06 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B2_B1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2274136, upper bound: 495.2385950
time: 1.31 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -212.5829620, 323.0835876, -225.7659760, 344.3264771, -556.9094238, 548.8495483
1: -237.4247437, 344.2901917, -252.2003479, 366.8069153, -604.2316895, 596.4905396
2: -241.4355774, 339.7105408, -256.4898987, 361.8593445, -603.2949219, 596.2004395
3: -290.1712646, 399.1027527, -308.3872681, 425.4865417, -715.6578369, 707.4899292
4: -264.3122253, 392.3148499, -281.1808777, 418.2401428, -682.5523682, 673.4957275

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B2_B2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2485776, upper bound: 495.2372411
time: 1.05 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B2_B2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2391607, upper bound: 495.2266103
time: 0.95 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B2_B2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A1_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2391607, upper bound: 495.2447791
time: 1.10 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -204.8689575, 310.3792725, -205.1396332, 312.5012817, -517.3701782, 515.5187378
1: -228.9616547, 330.9710388, -229.3643799, 333.4201050, -562.3817749, 560.3353271
2: -232.6103363, 326.3622131, -233.0310364, 328.6198425, -561.2301636, 559.3931885
3: -279.8240356, 383.7110596, -280.6290283, 386.7893372, -666.6134033, 664.3400879
4: -254.8099823, 377.3103027, -256.3842163, 379.5499268, -634.3599243, 633.6945190

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1_A1_B1_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.2046170, upper bound: 495.1088042
time: 0.94 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1_A1_B1_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_B2_A2_B1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2407630, upper bound: 495.2240181
time: 1.27 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.17 + 417.17 = 420.34 seconds
