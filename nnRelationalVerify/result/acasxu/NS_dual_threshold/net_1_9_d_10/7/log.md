## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 18.950919505144


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-11.6833382, 12.4738798, -11.6833382, 12.4738798, -24.1572151, 24.1572151)
1: (-91.6133957, 28.8711796, -91.6133957, 28.8711796, -120.4845734, 120.4845734)
2: (-47.9665756, 27.1552544, -47.9665756, 27.1552544, -75.1218262, 75.1218262)
3: (-62.7630501, 22.0255280, -62.7630501, 22.0255280, -84.7885742, 84.7885742)
4: (-33.5813065, 23.1921291, -33.5813065, 23.1921291, -56.7734375, 56.7734375)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.71 + 1.92 = 3.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -18.9524357, upper bound: 18.9524357

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9518763, upper bound: 18.9519298
time: 0.70 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9518763, upper bound: 18.9518763
time: 0.73 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.58 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 0, lower bound: -18.9518763, upper bound: 18.9519298
NS_B2, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 0, lower bound: -18.9518763, upper bound: 18.9518763

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -11.0580759, 11.8221369, -10.3281603, 11.0515471, -22.1096230, 22.1502972
1: -87.6119919, 27.2738838, -82.6208801, 25.4569778, -113.0689697, 109.8947449
2: -45.6439209, 25.7299995, -42.8500671, 24.0520840, -69.6960068, 68.5800629
3: -59.8711472, 20.8383999, -56.3412666, 19.4773655, -79.3485031, 77.1796646
4: -31.7946434, 21.9526691, -29.7730980, 20.4836941, -52.2783356, 51.7257690

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9518096, upper bound: 18.9518096
time: 0.98 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9518096, upper bound: 18.9518096
time: 0.73 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -11.3669538, 12.1905842, -17.4140701, 19.2958088, -30.6627579, 29.6046543
1: -88.8660049, 28.2998371, -136.1413116, 43.0780411, -131.9440308, 164.4411469
2: -46.7948532, 26.4937973, -70.8038712, 41.0104980, -87.8053513, 97.2976685
3: -61.1560364, 21.5206909, -92.8945236, 33.6828041, -94.8388367, 114.4152145
4: -32.8203812, 22.6683731, -49.5935059, 36.0196648, -68.8400421, 72.2618713

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515270, upper bound: 18.9518452
time: 1.59 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515056, upper bound: 18.9515056
time: 0.81 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.10 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 4.10
Output dim: 0, lower bound: -18.9518096, upper bound: 18.9518096
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 4.10
Output dim: 0, lower bound: -18.9518096, upper bound: 18.9518096
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 4.10
Output dim: 0, lower bound: -18.9515270, upper bound: 18.9518452
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 4.10
Output dim: 0, lower bound: -18.9515056, upper bound: 18.9515056

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -10.3281603, 11.0515471, -10.3281603, 11.0515471, -21.3797073, 21.3797073
1: -82.6208801, 25.4569778, -82.6208801, 25.4569778, -108.0778580, 108.0778580
2: -42.8500671, 24.0520840, -42.8500671, 24.0520840, -66.9021530, 66.9021530
3: -56.3412666, 19.4773655, -56.3412666, 19.4773655, -75.8186340, 75.8186340
4: -29.7730980, 20.4836941, -29.7730980, 20.4836941, -50.2567902, 50.2567902

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_B1

### Relational analysis result of NS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517490, upper bound: 18.9517464
time: 0.74 seconds

## Relational analysis of NS_B1_A1_B2

### Relational analysis result of NS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517852, upper bound: 18.9518387
time: 0.77 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -17.4140701, 19.2958088, -10.3281603, 11.0515471, -28.4656143, 29.6239681
1: -136.1413116, 43.0780411, -82.6208801, 25.4569778, -161.5982971, 125.6989212
2: -70.8038712, 41.0104980, -42.8500671, 24.0520840, -94.8559494, 83.8605652
3: -92.8945236, 33.6828041, -56.3412666, 19.4773655, -112.3718872, 90.0240707
4: -49.5935059, 36.0196648, -29.7730980, 20.4836941, -70.0771942, 65.7927628

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517066, upper bound: 18.9516561
time: 0.68 seconds

## Relational analysis of NS_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515777, upper bound: 18.9516609
time: 0.69 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -10.6524124, 11.4190826, -17.4140701, 19.2958088, -29.9482212, 28.8331470
1: -83.1773911, 26.5013103, -136.1413116, 43.0780411, -126.2554321, 162.6426239
2: -43.7759018, 24.8135796, -70.8038712, 41.0104980, -84.7863998, 95.6174469
3: -57.1881180, 20.2161560, -92.8945236, 33.6828041, -90.8709259, 113.1106796
4: -30.7249832, 21.2848549, -49.5935059, 36.0196648, -66.7446442, 70.8783493

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_A1

### Relational analysis result of NS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511704, upper bound: 18.9516867
time: 0.72 seconds

## Relational analysis of NS_B2_A1_A2

### Relational analysis result of NS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511704, upper bound: 18.9514743
time: 0.75 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -48.3735580, 50.7344971, -15.7999792, 17.4130554, -64.8651733, 65.9435272
1: -341.1856384, 117.5696945, -122.4180832, 38.9756927, -369.7427673, 239.9877625
2: -189.7698975, 108.7521591, -63.8048401, 37.0671844, -220.8140564, 172.2580109
3: -238.6354523, 88.3565750, -83.5949783, 30.3982105, -260.9320374, 171.9515533
4: -136.6828003, 94.8989105, -44.8380966, 32.5400124, -164.3392792, 139.6083832

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_A2_B1

### Relational analysis result of NS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515056, upper bound: 18.9515056
time: 0.94 seconds

## Relational analysis of NS_B2_A2_B2

### Relational analysis result of NS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515056, upper bound: 18.9515056
time: 0.96 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.59 seconds
NS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.59
Output dim: 0, lower bound: -18.9517490, upper bound: 18.9517464
NS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.59
Output dim: 0, lower bound: -18.9517852, upper bound: 18.9518387
NS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.59
Output dim: 0, lower bound: -18.9517066, upper bound: 18.9516561
NS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.59
Output dim: 0, lower bound: -18.9515777, upper bound: 18.9516609
NS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 3.59
Output dim: 0, lower bound: -18.9511704, upper bound: 18.9516867
NS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 3.59
Output dim: 0, lower bound: -18.9511704, upper bound: 18.9514743
NS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.59
Output dim: 0, lower bound: -18.9515056, upper bound: 18.9515056
NS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.59
Output dim: 0, lower bound: -18.9515056, upper bound: 18.9515056

## BFS NS instance: NS_B1_A1_B1

### Backsubstitution after applying NS history:
0: -9.9075117, 10.6043987, -9.5410004, 10.2219028, -20.1294136, 20.1453991
1: -79.7855530, 24.4337444, -77.3461533, 23.5448418, -103.3303986, 101.7798996
2: -41.2437630, 23.1033478, -39.8698807, 22.2895355, -63.5332947, 62.9732285
3: -54.3544998, 18.7060280, -52.6435242, 18.0404015, -72.3948975, 71.3495483
4: -28.6164150, 19.6349144, -27.6078606, 18.9090958, -47.5255127, 47.2427750

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A1_B1_B1

### Relational analysis result of NS_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517452, upper bound: 18.9515896
time: 0.76 seconds

## Relational analysis of NS_B1_A1_B1_B2

### Relational analysis result of NS_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9516413, upper bound: 18.9515896
time: 0.77 seconds

## BFS NS instance: NS_B1_A1_B2

### Backsubstitution after applying NS history:
0: -10.1348324, 10.8555441, -11.3833199, 12.4197531, -22.5545845, 22.2388649
1: -81.3478851, 24.9925842, -94.1245575, 28.1889305, -109.5368042, 119.1171341
2: -42.1167336, 23.6256142, -48.1099281, 26.8477879, -68.9644852, 71.7355423
3: -55.4342842, 19.1376820, -63.8559532, 21.8105049, -77.2447891, 82.9936371
4: -29.2381802, 20.1096287, -33.0945320, 22.8723297, -52.1105003, 53.2041588

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A1_B2_A1

### Relational analysis result of NS_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515778, upper bound: 18.9517610
time: 1.23 seconds

## Relational analysis of NS_B1_A1_B2_A2

### Relational analysis result of NS_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515778, upper bound: 18.9515778
time: 0.86 seconds

## BFS NS instance: NS_B1_A2_B1

### Backsubstitution after applying NS history:
0: -17.2129688, 19.0852509, -9.8210211, 10.6074114, -27.8203735, 28.9062691
1: -134.4648590, 42.5822067, -80.4366989, 24.3188705, -158.7837219, 123.0189056
2: -69.8880692, 40.5410080, -41.1475410, 23.0620880, -92.9501572, 81.6885529
3: -91.7399673, 33.3126793, -54.6445541, 18.7138824, -110.4538498, 87.9572220
4: -49.0164909, 35.6349297, -28.4959126, 19.6035500, -68.6200409, 64.1308441

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B1_B1

### Relational analysis result of NS_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517086, upper bound: 18.9514723
time: 0.85 seconds

## Relational analysis of NS_B1_A2_B1_B2

### Relational analysis result of NS_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517460, upper bound: 18.9515447
time: 0.87 seconds

## BFS NS instance: NS_B1_A2_B2

### Backsubstitution after applying NS history:
0: -17.0184059, 18.9062691, -9.3894138, 10.1268272, -27.1452312, 28.2956772
1: -133.6691437, 42.1163559, -76.7553940, 23.1929264, -156.8620758, 118.8717270
2: -69.3511124, 40.1370239, -39.4129219, 22.0109634, -91.3620682, 79.5499420
3: -91.1387405, 32.9916840, -52.1482773, 17.8483257, -108.9870529, 85.1399536
4: -48.4883652, 35.2777061, -27.1722641, 18.7199211, -67.2082748, 62.4499626

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B2_B1

### Relational analysis result of NS_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515478, upper bound: 18.9514723
time: 0.76 seconds

## Relational analysis of NS_B1_A2_B2_B2

### Relational analysis result of NS_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515514, upper bound: 18.9515447
time: 0.83 seconds

## BFS NS instance: NS_B2_A1_A1

### Backsubstitution after applying NS history:
0: -10.2026196, 11.0001450, -17.2129688, 19.0852509, -29.2878704, 28.2131100
1: -81.1365585, 25.4654350, -134.4648590, 42.5822067, -123.7187653, 159.9302826
2: -42.2413712, 23.9027443, -69.8880692, 40.5410080, -82.7823563, 93.7908173
3: -55.5976448, 19.5095367, -91.7399673, 33.3126793, -88.9103241, 111.2495041
4: -29.5800400, 20.4724350, -49.0164909, 35.6349297, -65.2149658, 69.4889221

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_B2_A1_A1_B1

### Relational analysis result of NS_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511670, upper bound: 18.9516589
time: 0.85 seconds

## Relational analysis of NS_B2_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B2_A1_A1_B1

### Relational analysis result of NS_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9508015, upper bound: 18.9514850
time: 0.81 seconds

## Relational analysis of NS_B2_A1_A1_B2

### Relational analysis result of NS_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9504997, upper bound: 18.9511232
time: 0.84 seconds

## BFS NS instance: NS_B2_A1_A2

### Backsubstitution after applying NS history:
0: -9.6426306, 10.4506483, -17.0211334, 18.9092369, -28.5518684, 27.4717827
1: -77.2980881, 23.9946995, -133.6826935, 42.1228027, -119.4208908, 157.6773987
2: -40.0057144, 22.6012058, -69.3599396, 40.1436195, -80.1493301, 91.9611435
3: -52.7640495, 18.4823303, -91.1485519, 32.9966011, -85.7606506, 109.6308823
4: -27.8986816, 19.4038105, -48.4952011, 35.2838326, -63.1825142, 67.8990097

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_B2_A1_A2_B1

### Relational analysis result of NS_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511669, upper bound: 18.9514497
time: 0.81 seconds

## Relational analysis of NS_B2_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B2_A1_A2_B1

### Relational analysis result of NS_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9507654, upper bound: 18.9509665
time: 2.02 seconds

## Relational analysis of NS_B2_A1_A2_B2

### Relational analysis result of NS_B2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9504957, upper bound: 18.9508943
time: 0.72 seconds

## BFS NS instance: NS_B2_A2_B1

### Backsubstitution after applying NS history:
0: -48.0336838, 50.3614693, -16.9493484, 18.7958755, -65.8561783, 66.6869583
1: -338.7916260, 116.7452621, -132.4742889, 41.8788414, -370.0747375, 248.5726471
2: -188.4772339, 107.9346313, -68.8888321, 39.9076653, -222.2165680, 176.1570892
3: -236.9860840, 87.7049408, -90.4129486, 32.7751465, -261.5127258, 178.1178284
4: -135.7440186, 94.1467667, -48.2513695, 35.0908470, -165.8807678, 142.0757446

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B2_A2_B1_B1

### Relational analysis result of NS_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514646, upper bound: 18.9513049
time: 1.00 seconds

## Relational analysis of NS_B2_A2_B1_B2

### Relational analysis result of NS_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514944, upper bound: 18.9514944
time: 0.81 seconds

## BFS NS instance: NS_B2_A2_B2

### Backsubstitution after applying NS history:
0: -49.8289833, 52.3599014, -54.1954308, 57.6227417, -105.3774796, 104.5835495
1: -351.4272461, 121.1203766, -382.9617004, 131.7783813, -469.3549194, 489.3827820
2: -195.2707062, 112.3526306, -211.7870789, 122.8398056, -309.3764038, 315.4094238
3: -245.6848907, 91.2153625, -267.1886292, 100.0899658, -335.7657471, 348.2563782
4: -140.6742554, 98.2709503, -152.5055237, 108.4809265, -242.2256775, 244.2824097

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_B2_A1

### Relational analysis result of NS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511544, upper bound: 18.9513850
time: 0.80 seconds

## Relational analysis of NS_B2_A2_B2_A2

### Relational analysis result of NS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511528, upper bound: 18.9511528
time: 0.86 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.08 seconds
NS_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -18.9517452, upper bound: 18.9515896
NS_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -18.9516413, upper bound: 18.9515896
NS_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -18.9515778, upper bound: 18.9517610
NS_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -18.9515778, upper bound: 18.9515778
NS_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -18.9517086, upper bound: 18.9514723
NS_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -18.9517460, upper bound: 18.9515447
NS_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -18.9515478, upper bound: 18.9514723
NS_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -18.9515514, upper bound: 18.9515447
NS_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -18.9508015, upper bound: 18.9514850
NS_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -18.9504997, upper bound: 18.9511232
NS_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -18.9507654, upper bound: 18.9509665
NS_B2_A1_A2_B2, status: Status.VERIFIED, split count: 4, time: 4.08
Output dim: 0, lower bound: -18.9504957, upper bound: 18.9508943
NS_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -18.9514646, upper bound: 18.9513049
NS_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -18.9514944, upper bound: 18.9514944
NS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -18.9511544, upper bound: 18.9513850
NS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -18.9511528, upper bound: 18.9511528

## BFS NS instance: NS_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -9.7142191, 10.4082336, -9.2110405, 9.9745607, -19.6887760, 19.6192741
1: -78.2465286, 23.9614830, -76.3690186, 22.8215199, -101.0680466, 100.3305054
2: -40.4033089, 22.6608028, -38.8359337, 21.6892929, -62.0925980, 61.4967346
3: -53.2783966, 18.3622227, -51.7815933, 17.6024361, -70.8808289, 70.1438141
4: -28.0444603, 19.2795620, -26.7976322, 18.3903713, -46.4348297, 46.0771866

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A1_B1_B1_A1

### Relational analysis result of NS_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9516413, upper bound: 18.9515896
time: 0.73 seconds

## Relational analysis of NS_B1_A1_B1_B1_A2

### Relational analysis result of NS_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9516413, upper bound: 18.9515896
time: 0.78 seconds

## BFS NS instance: NS_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -9.4807196, 10.1908627, -8.5857410, 9.2967005, -18.7774181, 18.7766037
1: -77.1531830, 23.3994198, -71.3606720, 21.2869167, -98.4400635, 94.7600937
2: -39.7012711, 22.1851959, -36.3787956, 20.2452316, -59.9464951, 58.5639877
3: -52.4695206, 17.9722214, -48.3730583, 16.4173374, -68.8868561, 66.3452759
4: -27.4357376, 18.8469048, -24.9768715, 17.1562386, -44.5919762, 43.8237762

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A1_B1_B2_A1

### Relational analysis result of NS_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9516413, upper bound: 18.9515896
time: 0.69 seconds

## Relational analysis of NS_B1_A1_B1_B2_A2

### Relational analysis result of NS_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9516413, upper bound: 18.9515896
time: 0.69 seconds

## BFS NS instance: NS_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -9.6540794, 10.4347181, -11.1627655, 12.1892815, -21.8433609, 21.5974827
1: -79.3318558, 23.9139748, -92.3871231, 27.6228523, -106.9547043, 116.3010941
2: -40.5086136, 22.6908417, -47.1222954, 26.3352623, -66.8438721, 69.8131409
3: -53.8613892, 18.4144821, -62.6433334, 21.4003105, -75.2617035, 81.0578156
4: -28.0318661, 19.2741928, -32.4451828, 22.4504395, -50.4823036, 51.7193756

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A1_B2_A1_B1

### Relational analysis result of NS_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515778, upper bound: 18.9515778
time: 1.20 seconds

## Relational analysis of NS_B1_A1_B2_A1_B2

### Relational analysis result of NS_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515778, upper bound: 18.9515778
time: 0.98 seconds

## BFS NS instance: NS_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -9.1871710, 9.9213552, -10.9891129, 12.0381832, -21.2253532, 20.9104671
1: -75.4456024, 22.7175980, -91.7071228, 27.2714214, -102.7170258, 114.4247208
2: -38.6633911, 21.5696259, -46.6865997, 26.0075703, -64.6709595, 68.2562256
3: -51.2161713, 17.4949360, -62.1220093, 21.1515484, -72.3677139, 79.6169434
4: -26.6209526, 18.3268528, -32.0371399, 22.1488953, -48.7698402, 50.3639793

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A1_B2_A2_B1

### Relational analysis result of NS_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515778, upper bound: 18.9515778
time: 0.77 seconds

## Relational analysis of NS_B1_A1_B2_A2_B2

### Relational analysis result of NS_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515778, upper bound: 18.9515778
time: 1.37 seconds

## BFS NS instance: NS_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -16.8571892, 18.7137985, -9.2110405, 9.9745607, -26.8317471, 27.9248390
1: -132.0391083, 41.7055626, -76.3690186, 22.8215199, -154.8606110, 118.0745850
2: -68.5282211, 39.7350807, -38.8359337, 21.6892929, -90.2175140, 78.5710144
3: -90.0532990, 32.6619072, -51.7815933, 17.6024361, -107.6557312, 84.4434891
4: -48.0134277, 34.9398537, -26.7976322, 18.3903713, -66.4037933, 61.7374802

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A2_B1_B1_A1

### Relational analysis result of NS_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517001, upper bound: 18.9514667
time: 1.12 seconds

## Relational analysis of NS_B1_A2_B1_B1_A2

### Relational analysis result of NS_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517001, upper bound: 18.9514578
time: 0.77 seconds

## BFS NS instance: NS_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -17.0241241, 18.8767185, -11.0636187, 12.1841145, -29.2082367, 29.9403343
1: -133.1528625, 42.1220589, -92.9218903, 27.4736214, -160.6264648, 135.0439453
2: -69.1795273, 40.1034813, -47.0095863, 26.2644234, -95.4439545, 87.1130676
3: -90.8333206, 32.9569511, -62.8485527, 21.3922367, -112.2255402, 95.8055038
4: -48.4977417, 35.2345314, -32.2702484, 22.4078445, -70.9055862, 67.5047684

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_B1_B2_A1

### Relational analysis result of NS_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515902, upper bound: 18.9513981
time: 0.76 seconds

## Relational analysis of NS_B1_A2_B1_B2_A2

### Relational analysis result of NS_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514118, upper bound: 18.9511930
time: 0.73 seconds

## BFS NS instance: NS_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -16.6520824, 18.5249348, -8.5857410, 9.2967005, -25.9487782, 27.1106758
1: -131.1652985, 41.2191544, -71.3606720, 21.2869167, -152.4522095, 112.5798187
2: -67.9589081, 39.3209877, -36.3787956, 20.2452316, -88.2041321, 75.6997833
3: -89.4016647, 32.3235664, -48.3730583, 16.4173374, -105.8190002, 80.6966248
4: -47.4644814, 34.5694695, -24.9768715, 17.1562386, -64.6207199, 59.5463409

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A2_B2_B1_A1

### Relational analysis result of NS_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515384, upper bound: 18.9514667
time: 0.72 seconds

## Relational analysis of NS_B1_A2_B2_B1_A2

### Relational analysis result of NS_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515384, upper bound: 18.9514578
time: 0.67 seconds

## BFS NS instance: NS_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -16.8133507, 18.6798534, -10.5182152, 11.5731764, -28.3865242, 29.1980686
1: -132.2351227, 41.6163101, -88.7561264, 26.1512699, -158.3863678, 130.3724365
2: -68.5770721, 39.6633110, -44.9450226, 24.9894428, -93.5665131, 84.6083221
3: -90.1483002, 32.6041298, -60.0080109, 20.3411846, -110.4894791, 92.6120987
4: -47.9252319, 34.8439102, -30.7488251, 21.2645836, -69.1898193, 65.5927353

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_B2_B2_A1

### Relational analysis result of NS_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512103, upper bound: 18.9513981
time: 0.86 seconds

## Relational analysis of NS_B1_A2_B2_B2_A2

### Relational analysis result of NS_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512049, upper bound: 18.9511930
time: 0.78 seconds

## BFS NS instance: NS_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -10.2026196, 11.0001450, -16.9201336, 18.7835770, -28.9861965, 27.9202785
1: -81.1365585, 25.4654350, -132.6551819, 41.8811455, -123.0177002, 158.1206207
2: -42.2413712, 23.9027443, -68.8311996, 39.8962555, -82.1376190, 92.7339478
3: -55.5976448, 19.5095367, -90.4635544, 32.8015366, -88.3991852, 109.9730911
4: -29.5800400, 20.4724350, -48.2103157, 35.0624886, -64.6425323, 68.6827545

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_A1_B1_B1

### Relational analysis result of NS_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9507970, upper bound: 18.9514850
time: 0.88 seconds

## Relational analysis of NS_B2_A1_A1_B1_B2

### Relational analysis result of NS_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9508015, upper bound: 18.9514850
time: 0.86 seconds

## BFS NS instance: NS_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -10.1632013, 10.9500790, -16.7307110, 18.5362740, -28.6994724, 27.6807899
1: -80.7509766, 25.3672752, -129.8013000, 41.3716545, -122.1226349, 155.1685638
2: -42.0703850, 23.8040047, -67.7051544, 39.3646889, -81.4350662, 91.5091553
3: -55.3507767, 19.4250984, -88.6639175, 32.3560753, -87.7068481, 108.0890198
4: -29.4617786, 20.3839436, -47.5472832, 34.6662369, -64.1280136, 67.9312286

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B2_A1_A1_B2_B1

### Relational analysis result of NS_B2_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9501916, upper bound: 18.9507261
time: 0.90 seconds

## Relational analysis of NS_B2_A1_A1_B2_B2

### Relational analysis result of NS_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9504921, upper bound: 18.9511217
time: 0.85 seconds

## BFS NS instance: NS_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -9.6426306, 10.4506483, -16.7268715, 18.6061382, -28.2487679, 27.1775208
1: -77.2980881, 23.9946995, -131.8437347, 41.4428406, -118.7409286, 155.8383942
2: -40.0057144, 22.6012058, -68.2936172, 39.5050011, -79.5107117, 90.8948212
3: -52.7640495, 18.4823303, -89.8541031, 32.4894753, -85.2535248, 108.3364334
4: -27.8986816, 19.4038105, -47.6858177, 34.7145538, -62.6132317, 67.0896301

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_A2_B1_B1

### Relational analysis result of NS_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9507605, upper bound: 18.9509665
time: 0.78 seconds

## Relational analysis of NS_B2_A1_A2_B1_B2

### Relational analysis result of NS_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9507654, upper bound: 18.9509665
time: 0.76 seconds

## BFS NS instance: NS_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -47.7184486, 50.0011024, -15.7599220, 17.4984741, -64.2291489, 65.1199875
1: -336.1123657, 115.9493942, -122.3157501, 38.8930779, -364.3532715, 237.4657135
2: -187.1389771, 107.1439667, -63.9115372, 37.0650978, -217.9717865, 170.2528229
3: -235.1795654, 87.0731506, -83.5661087, 30.4714451, -257.3244629, 170.6392517
4: -134.8140869, 93.4726257, -44.7390747, 32.7124863, -162.5528259, 137.7845459

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_B1_B1_B1

### Relational analysis result of NS_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515081, upper bound: 18.9507588
time: 0.72 seconds

## Relational analysis of NS_B2_A2_B1_B1_B2

### Relational analysis result of NS_B2_A2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9507412, upper bound: 18.9505126
time: 0.69 seconds

## BFS NS instance: NS_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -47.8338318, 50.1353149, -16.8161449, 18.6487465, -65.4927521, 66.3168793
1: -337.4182434, 116.2596283, -131.6379089, 41.5574112, -368.3253479, 247.1891937
2: -187.7334900, 107.4540634, -68.4160919, 39.6029587, -221.0983429, 175.1613770
3: -236.0416260, 87.3171005, -89.8289642, 32.5277786, -260.2729187, 177.1460419
4: -135.2015686, 93.6941910, -47.8969650, 34.8093872, -164.9949951, 141.2336578

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_B1_B2_B1

### Relational analysis result of NS_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9516867, upper bound: 18.9511704
time: 0.79 seconds

## Relational analysis of NS_B2_A2_B1_B2_B2

### Relational analysis result of NS_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514743, upper bound: 18.9511704
time: 0.73 seconds

## BFS NS instance: NS_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -49.2961617, 51.8081856, -53.9746971, 57.3931046, -104.5459518, 103.7753983
1: -348.1985474, 119.8371582, -381.3567200, 131.2411194, -465.2501221, 486.2454834
2: -193.2834930, 111.1613693, -210.8939972, 122.3300171, -306.6286926, 313.1592407
3: -243.3537598, 90.2618637, -266.0723572, 99.6916199, -332.7619019, 346.0326538
4: -139.2274780, 97.2031631, -151.8725891, 108.0631332, -240.1582794, 242.4898224

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_B2_A1_B1

### Relational analysis result of NS_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511528, upper bound: 18.9511528
time: 0.70 seconds

## Relational analysis of NS_B2_A2_B2_A1_B2

### Relational analysis result of NS_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511528, upper bound: 18.9511528
time: 0.81 seconds

## BFS NS instance: NS_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -48.9295502, 51.5034065, -53.7897491, 57.2326164, -104.1152420, 103.3267822
1: -345.8834839, 118.8759384, -380.3610535, 130.7750702, -462.5991516, 484.5331421
2: -191.7705536, 110.3737793, -210.1450500, 121.9407120, -305.1115723, 311.8263550
3: -241.6047211, 89.6515427, -265.2525330, 99.3725891, -330.9397888, 344.7973633
4: -138.1627808, 96.6145477, -151.3624725, 107.7300873, -239.0725250, 241.5107269

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_B2_A2_B1

### Relational analysis result of NS_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511528, upper bound: 18.9511528
time: 0.81 seconds

## Relational analysis of NS_B2_A2_B2_A2_B2

### Relational analysis result of NS_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511528, upper bound: 18.9511528
time: 0.77 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.43 seconds
NS_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -18.9516413, upper bound: 18.9515896
NS_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -18.9516413, upper bound: 18.9515896
NS_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -18.9516413, upper bound: 18.9515896
NS_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -18.9516413, upper bound: 18.9515896
NS_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -18.9515778, upper bound: 18.9515778
NS_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -18.9515778, upper bound: 18.9515778
NS_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -18.9515778, upper bound: 18.9515778
NS_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -18.9515778, upper bound: 18.9515778
NS_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -18.9517001, upper bound: 18.9514667
NS_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -18.9517001, upper bound: 18.9514578
NS_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -18.9515902, upper bound: 18.9513981
NS_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -18.9514118, upper bound: 18.9511930
NS_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -18.9515384, upper bound: 18.9514667
NS_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -18.9515384, upper bound: 18.9514578
NS_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -18.9512103, upper bound: 18.9513981
NS_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -18.9512049, upper bound: 18.9511930
NS_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -18.9507970, upper bound: 18.9514850
NS_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -18.9508015, upper bound: 18.9514850
NS_B2_A1_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 3.43
Output dim: 0, lower bound: -18.9501916, upper bound: 18.9507261
NS_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -18.9504921, upper bound: 18.9511217
NS_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -18.9507605, upper bound: 18.9509665
NS_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -18.9507654, upper bound: 18.9509665
NS_B2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -18.9515081, upper bound: 18.9507588
NS_B2_A2_B1_B1_B2, status: Status.VERIFIED, split count: 5, time: 3.43
Output dim: 0, lower bound: -18.9507412, upper bound: 18.9505126
NS_B2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -18.9516867, upper bound: 18.9511704
NS_B2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -18.9514743, upper bound: 18.9511704
NS_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -18.9511528, upper bound: 18.9511528
NS_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -18.9511528, upper bound: 18.9511528
NS_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -18.9511528, upper bound: 18.9511528
NS_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -18.9511528, upper bound: 18.9511528

## BFS NS instance: NS_B1_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -9.4966640, 10.2702999, -9.2110405, 9.9745607, -19.4712219, 19.4813404
1: -78.2795029, 23.5250149, -76.3690186, 22.8215199, -101.1010132, 99.8940277
2: -39.9114876, 22.3295212, -38.8359337, 21.6892929, -61.6007805, 61.1654549
3: -53.1255913, 18.1230240, -51.7815933, 17.6024361, -70.7280197, 69.9046173
4: -27.5965233, 18.9551182, -26.7976322, 18.3903713, -45.9868851, 45.7527466

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_A1_B1_B1_A1_B1

### Relational analysis result of NS_B1_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517439, upper bound: 18.9514176
time: 0.70 seconds

## Relational analysis of NS_B1_A1_B1_B1_A1_B2

### Relational analysis result of NS_B1_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517439, upper bound: 18.9515790
time: 0.72 seconds

## BFS NS instance: NS_B1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -8.9571800, 9.6799469, -9.2110405, 9.9745607, -18.9317341, 18.8909855
1: -73.8682709, 22.1722736, -76.3690186, 22.8215199, -96.6897888, 98.5412903
2: -37.7859039, 21.0613575, -38.8359337, 21.6892929, -59.4751968, 59.8972931
3: -50.1258888, 17.0795021, -51.7815933, 17.6024361, -67.7283249, 68.8610992
4: -25.9968891, 17.8761425, -26.7976322, 18.3903713, -44.3872528, 44.6737709

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_A1_B1_B1_A2_B1

### Relational analysis result of NS_B1_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517439, upper bound: 18.9514176
time: 0.77 seconds

## Relational analysis of NS_B1_A1_B1_B1_A2_B2

### Relational analysis result of NS_B1_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517439, upper bound: 18.9515790
time: 0.74 seconds

## BFS NS instance: NS_B1_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -9.4966640, 10.2702999, -8.5857410, 9.2967005, -18.7933636, 18.8560410
1: -78.2795029, 23.5250149, -71.3606720, 21.2869167, -99.5663910, 94.8856812
2: -39.9114876, 22.3295212, -36.3787956, 20.2452316, -60.1567154, 58.7083130
3: -53.1255913, 18.1230240, -48.3730583, 16.4173374, -69.5429153, 66.4960785
4: -27.5965233, 18.9551182, -24.9768715, 17.1562386, -44.7527618, 43.9319916

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_A1_B1_B2_A1_B1

### Relational analysis result of NS_B1_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513517, upper bound: 18.9513900
time: 0.74 seconds

## Relational analysis of NS_B1_A1_B1_B2_A1_B2

### Relational analysis result of NS_B1_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9516339, upper bound: 18.9515790
time: 0.77 seconds

## BFS NS instance: NS_B1_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -8.9571800, 9.6799469, -8.5857410, 9.2967005, -18.2538757, 18.2656841
1: -73.8682709, 22.1722736, -71.3606720, 21.2869167, -95.1551743, 93.5329285
2: -37.7859039, 21.0613575, -36.3787956, 20.2452316, -58.0311356, 57.4401512
3: -50.1258888, 17.0795021, -48.3730583, 16.4173374, -66.5432129, 65.4525604
4: -25.9968891, 17.8761425, -24.9768715, 17.1562386, -43.1531219, 42.8530121

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_A1_B1_B2_A2_B1

### Relational analysis result of NS_B1_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513517, upper bound: 18.9513900
time: 0.76 seconds

## Relational analysis of NS_B1_A1_B1_B2_A2_B2

### Relational analysis result of NS_B1_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9516339, upper bound: 18.9515790
time: 1.25 seconds

## BFS NS instance: NS_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -9.6540794, 10.4347181, -11.0636187, 12.1841145, -21.8381939, 21.4983368
1: -79.3318558, 23.9139748, -92.9218903, 27.4736214, -106.8054733, 116.8358612
2: -40.5086136, 22.6908417, -47.0095863, 26.2644234, -66.7730408, 69.7004242
3: -53.8613892, 18.4144821, -62.8485527, 21.3922367, -75.2536240, 81.2630310
4: -28.0318661, 19.2741928, -32.2702484, 22.4078445, -50.4397087, 51.5444412

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_A1_B2_A1_B1_B1

### Relational analysis result of NS_B1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512602, upper bound: 18.9514627
time: 0.71 seconds

## Relational analysis of NS_B1_A1_B2_A1_B1_B2

### Relational analysis result of NS_B1_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515720, upper bound: 18.9517601
time: 0.70 seconds

## BFS NS instance: NS_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -9.6540794, 10.4347181, -10.5182152, 11.5731764, -21.2272568, 20.9529343
1: -79.3318558, 23.9139748, -88.7561264, 26.1512699, -105.4831085, 112.6701050
2: -40.5086136, 22.6908417, -44.9450226, 24.9894428, -65.4980545, 67.6358566
3: -53.8613892, 18.4144821, -60.0080109, 20.3411846, -74.2025757, 78.4224854
4: -28.0318661, 19.2741928, -30.7488251, 21.2645836, -49.2964478, 50.0230179

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_A1_B2_A1_B2_B1

### Relational analysis result of NS_B1_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512602, upper bound: 18.9514627
time: 0.72 seconds

## Relational analysis of NS_B1_A1_B2_A1_B2_B2

### Relational analysis result of NS_B1_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515720, upper bound: 18.9517601
time: 0.71 seconds

## BFS NS instance: NS_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -9.1871710, 9.9213552, -11.0636187, 12.1841145, -21.3712845, 20.9849720
1: -75.4456024, 22.7175980, -92.9218903, 27.4736214, -102.9192200, 115.6394806
2: -38.6633911, 21.5696259, -47.0095863, 26.2644234, -64.9278107, 68.5792084
3: -51.2161713, 17.4949360, -62.8485527, 21.3922367, -72.6083984, 80.3434906
4: -26.6209526, 18.3268528, -32.2702484, 22.4078445, -49.0287933, 50.5970879

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_A1_B2_A2_B1_B1

### Relational analysis result of NS_B1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512602, upper bound: 18.9513843
time: 0.68 seconds

## Relational analysis of NS_B1_A1_B2_A2_B1_B2

### Relational analysis result of NS_B1_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515720, upper bound: 18.9515720
time: 0.77 seconds

## BFS NS instance: NS_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -9.1871710, 9.9213552, -10.5182152, 11.5731764, -20.7603455, 20.4395676
1: -75.4456024, 22.7175980, -88.7561264, 26.1512699, -101.5968704, 111.4737244
2: -38.6633911, 21.5696259, -44.9450226, 24.9894428, -63.6528320, 66.5146484
3: -51.2161713, 17.4949360, -60.0080109, 20.3411846, -71.5573578, 77.5029221
4: -26.6209526, 18.3268528, -30.7488251, 21.2645836, -47.8855324, 49.0756683

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_A1_B2_A2_B2_B1

### Relational analysis result of NS_B1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512602, upper bound: 18.9513843
time: 0.72 seconds

## Relational analysis of NS_B1_A1_B2_A2_B2_B2

### Relational analysis result of NS_B1_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515720, upper bound: 18.9515720
time: 0.73 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -16.7740650, 18.6175041, -9.2110405, 9.9745607, -26.7486229, 27.8285446
1: -131.3912506, 41.4975471, -76.3690186, 22.8215199, -154.2127380, 117.8665543
2: -68.1971588, 39.5344696, -38.8359337, 21.6892929, -89.8864517, 78.3704071
3: -89.6147842, 32.4974747, -51.7815933, 17.6024361, -107.2172241, 84.2790604
4: -47.7793007, 34.7612991, -26.7976322, 18.3903713, -66.1696625, 61.5589256

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_B1_B1_A1_A1

### Relational analysis result of NS_B1_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517001, upper bound: 18.9514667
time: 0.71 seconds

## Relational analysis of NS_B1_A2_B1_B1_A1_A2

### Relational analysis result of NS_B1_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517001, upper bound: 18.9514667
time: 0.90 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -16.8220310, 18.6687088, -9.2110405, 9.9745607, -26.7965851, 27.8797493
1: -131.5673065, 41.6107254, -76.3690186, 22.8215199, -154.3888245, 117.9797440
2: -68.3342285, 39.6290092, -38.8359337, 21.6892929, -90.0235214, 78.4649277
3: -89.7509232, 32.5845871, -51.7815933, 17.6024361, -107.3533478, 84.3661804
4: -47.8952560, 34.8633270, -26.7976322, 18.3903713, -66.2856140, 61.6609573

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_B1_B1_A2_A1

### Relational analysis result of NS_B1_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517001, upper bound: 18.9514578
time: 0.70 seconds

## Relational analysis of NS_B1_A2_B1_B1_A2_A2

### Relational analysis result of NS_B1_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517001, upper bound: 18.9514578
time: 0.76 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -16.7289295, 18.5726070, -11.0636187, 12.1841145, -28.9130440, 29.6362267
1: -131.3232269, 41.4167099, -92.9218903, 27.4736214, -158.7968292, 134.3385925
2: -68.1127548, 39.4526787, -47.0095863, 26.2644234, -94.3771744, 86.4622650
3: -89.5436630, 32.4410362, -62.8485527, 21.3922367, -110.9358902, 95.2895889
4: -47.6844292, 34.6568909, -32.2702484, 22.4078445, -70.0922699, 66.9271240

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B1_A2_B1_B2_A1_A1

### Relational analysis result of NS_B1_A2_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9506431, upper bound: 18.9502601
time: 0.78 seconds

## Relational analysis of NS_B1_A2_B1_B2_A1_A2

### Relational analysis result of NS_B1_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9506431, upper bound: 18.9513819
time: 0.76 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -16.5517235, 18.3383255, -11.0283537, 12.1403561, -28.6920795, 29.3666801
1: -128.5610504, 40.9361572, -92.5713120, 27.3807812, -155.9418335, 133.5074768
2: -67.0270767, 38.9505043, -46.8399239, 26.1737003, -93.2007599, 85.7904205
3: -87.8019257, 32.0179482, -62.6170387, 21.3170795, -109.1190033, 94.6349716
4: -47.0568352, 34.2857018, -32.1591377, 22.3303127, -69.3871460, 66.4448395

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B1_A2_B1_B2_A2_A1

### Relational analysis result of NS_B1_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511453, upper bound: 18.9509827
time: 0.77 seconds

## Relational analysis of NS_B1_A2_B1_B2_A2_A2

### Relational analysis result of NS_B1_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511453, upper bound: 18.9511830
time: 0.85 seconds

## BFS NS instance: NS_B1_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -16.5683250, 18.4283581, -8.5857410, 9.2967005, -25.8650188, 27.0140991
1: -130.5123444, 41.0102234, -71.3606720, 21.2869167, -151.7992554, 112.3708878
2: -67.6274643, 39.1205940, -36.3787956, 20.2452316, -87.8726883, 75.4993896
3: -88.9596710, 32.1588631, -48.3730583, 16.4173374, -105.3770065, 80.5319138
4: -47.2277794, 34.3911667, -24.9768715, 17.1562386, -64.3840179, 59.3680382

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B2_B1_A1_A1

### Relational analysis result of NS_B1_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515087, upper bound: 18.9513244
time: 0.77 seconds

## Relational analysis of NS_B1_A2_B2_B1_A1_A2

### Relational analysis result of NS_B1_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514740, upper bound: 18.9514137
time: 0.69 seconds

## BFS NS instance: NS_B1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -16.6155014, 18.4771652, -8.5857410, 9.2967005, -25.9121990, 27.0629063
1: -130.6765594, 41.1182060, -71.3606720, 21.2869167, -151.9634705, 112.4788666
2: -67.7580185, 39.2104912, -36.3787956, 20.2452316, -88.0032425, 75.5892868
3: -89.0882492, 32.2415619, -48.3730583, 16.4173374, -105.5055847, 80.6146164
4: -47.3426666, 34.4884377, -24.9768715, 17.1562386, -64.4989014, 59.4653091

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B2_B1_A2_A1

### Relational analysis result of NS_B1_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515087, upper bound: 18.9513205
time: 0.76 seconds

## Relational analysis of NS_B1_A2_B2_B1_A2_A2

### Relational analysis result of NS_B1_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514723, upper bound: 18.9514092
time: 0.71 seconds

## BFS NS instance: NS_B1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -16.5202560, 18.3773422, -10.5182152, 11.5731764, -28.0934296, 28.8955574
1: -130.3952179, 40.9408073, -88.7561264, 26.1512699, -156.5464630, 129.6969299
2: -67.5156403, 39.0260468, -44.9450226, 24.9894428, -92.5050812, 83.9710617
3: -88.8553314, 32.0997353, -60.0080109, 20.3411846, -109.1965179, 92.1077194
4: -47.1219406, 34.2753639, -30.7488251, 21.2645836, -68.3865204, 65.0241852

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_B1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B1_A2_B2_B2_A1_A1

### Relational analysis result of NS_B1_A2_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9506162, upper bound: 18.9502601
time: 0.74 seconds

## Relational analysis of NS_B1_A2_B2_B2_A1_A2

### Relational analysis result of NS_B1_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512083, upper bound: 18.9513819
time: 0.82 seconds

## BFS NS instance: NS_B1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -16.3368759, 18.1371288, -10.4842367, 11.5324278, -27.8693047, 28.6213646
1: -127.5943451, 40.4123840, -88.4207001, 26.0651512, -153.6594696, 128.8330841
2: -66.3490829, 38.4932213, -44.7860527, 24.9050179, -91.2540894, 83.2792740
3: -87.0780487, 31.6560116, -59.7870712, 20.2714138, -107.3494644, 91.4430847
4: -46.4821587, 33.8797760, -30.6447449, 21.1933403, -67.6754990, 64.5245209

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_B1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B1_A2_B2_B2_A2_A1

### Relational analysis result of NS_B1_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510912, upper bound: 18.9509827
time: 1.11 seconds

## Relational analysis of NS_B1_A2_B2_B2_A2_A2

### Relational analysis result of NS_B1_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512039, upper bound: 18.9511830
time: 0.78 seconds

## BFS NS instance: NS_B2_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -10.2026196, 11.0001450, -16.8371754, 18.6869144, -28.8895340, 27.8373146
1: -81.1365585, 25.4654350, -132.0029297, 41.6735649, -122.8101196, 157.4683533
2: -42.2413712, 23.9027443, -68.4992065, 39.6953659, -81.9366913, 92.4019470
3: -55.5976448, 19.5095367, -90.0227585, 32.6368370, -88.2344818, 109.5322952
4: -29.5800400, 20.4724350, -47.9755745, 34.8836746, -64.4637146, 68.4480133

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_A1_B1_B1_B1

### Relational analysis result of NS_B2_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9507970, upper bound: 18.9514850
time: 0.83 seconds

## Relational analysis of NS_B2_A1_A1_B1_B1_B2

### Relational analysis result of NS_B2_A1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9507970, upper bound: 18.9514850
time: 0.83 seconds

## BFS NS instance: NS_B2_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -10.2026196, 11.0001450, -16.8855724, 18.7392139, -28.9418335, 27.8857174
1: -81.1365585, 25.4654350, -132.1930542, 41.7877693, -122.9243240, 157.6584778
2: -42.2413712, 23.9027443, -68.6414719, 39.7918091, -82.0331650, 92.5442123
3: -55.5976448, 19.5095367, -90.1674423, 32.7250595, -88.3227081, 109.6769791
4: -29.5800400, 20.4724350, -48.0943260, 34.9875870, -64.5676270, 68.5667572

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_A1_B1_B2_B1

### Relational analysis result of NS_B2_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9508015, upper bound: 18.9514850
time: 0.74 seconds

## Relational analysis of NS_B2_A1_A1_B1_B2_B2

### Relational analysis result of NS_B2_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9508015, upper bound: 18.9514850
time: 0.73 seconds

## BFS NS instance: NS_B2_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -10.1318464, 10.9177208, -16.5838146, 18.3775921, -28.5094376, 27.5015297
1: -80.5541840, 25.2932720, -128.9253387, 41.0192947, -121.5734787, 154.2186127
2: -41.9504547, 23.7347755, -67.1838989, 39.0367508, -80.9872055, 90.9186630
3: -55.2103462, 19.3708324, -88.0454865, 32.0878677, -87.2982101, 107.4163208
4: -29.3772621, 20.3219624, -47.1610260, 34.3637924, -63.7410545, 67.4829788

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B2_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_A1_B2_B2_B1

### Relational analysis result of NS_B2_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9503661, upper bound: 18.9510133
time: 0.95 seconds

## Relational analysis of NS_B2_A1_A1_B2_B2_B2

### Relational analysis result of NS_B2_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9504921, upper bound: 18.9511217
time: 0.87 seconds

## BFS NS instance: NS_B2_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -9.6426306, 10.4506483, -16.6443024, 18.5102978, -28.1529274, 27.0949516
1: -77.2980881, 23.9946995, -131.1936951, 41.2369499, -118.5350342, 155.1883698
2: -40.0057144, 22.6012058, -67.9648132, 39.3062477, -79.3119507, 90.5660172
3: -52.7640495, 18.4823303, -89.4153595, 32.3260918, -85.0901337, 107.8976898
4: -27.8986816, 19.4038105, -47.4528122, 34.5374680, -62.4361496, 66.8566208

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B2_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_A2_B1_B1_B1

### Relational analysis result of NS_B2_A1_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9505778, upper bound: 18.9508942
time: 0.78 seconds

## Relational analysis of NS_B2_A1_A2_B1_B1_B2

### Relational analysis result of NS_B2_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9507208, upper bound: 18.9509279
time: 0.80 seconds

## BFS NS instance: NS_B2_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -9.6426306, 10.4506483, -16.6878719, 18.5563488, -28.1989784, 27.1385193
1: -77.2980881, 23.9946995, -131.3449249, 41.3372498, -118.6353149, 155.3396149
2: -40.0057144, 22.6012058, -68.0844040, 39.3898048, -79.3955154, 90.6856003
3: -52.7640495, 18.4823303, -89.5325394, 32.4043884, -85.1684418, 108.0148697
4: -27.8986816, 19.4038105, -47.5591049, 34.6296844, -62.5283661, 66.9629135

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B2_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_A2_B1_B2_B1

### Relational analysis result of NS_B2_A1_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9506660, upper bound: 18.9508942
time: 0.76 seconds

## Relational analysis of NS_B2_A1_A2_B1_B2_B2

### Relational analysis result of NS_B2_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9507273, upper bound: 18.9509279
time: 0.83 seconds

## BFS NS instance: NS_B2_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -47.1845360, 49.4163589, -15.3291674, 17.0978394, -63.2540169, 64.0599289
1: -332.4113770, 114.6516190, -119.9082947, 37.8919601, -359.3556824, 233.5250549
2: -185.1062012, 105.8947601, -62.2990036, 36.1622429, -214.8143158, 167.2521973
3: -232.6130219, 86.0612183, -81.7945328, 29.7906189, -253.8606567, 167.8373871
4: -133.3425140, 92.3301239, -43.5816612, 31.9553642, -160.1754303, 135.4127350

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_B1_B1_B1_A1

### Relational analysis result of NS_B2_A2_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9507412, upper bound: 18.9505126
time: 0.76 seconds

## Relational analysis of NS_B2_A2_B1_B1_B1_A2

### Relational analysis result of NS_B2_A2_B1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9507412, upper bound: 18.9505126
time: 0.78 seconds

## BFS NS instance: NS_B2_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -47.3444214, 49.6022568, -16.5459156, 18.4468384, -64.7605896, 65.4824219
1: -334.0079041, 115.0704041, -130.5139160, 40.9471626, -364.0585632, 244.7073059
2: -185.8607635, 106.2930145, -67.4658813, 39.0936546, -218.5264893, 172.9313354
3: -233.6818848, 86.3968124, -88.9198608, 32.1694412, -257.3607178, 175.3166809
4: -133.8470459, 92.6496735, -47.1886597, 34.4116898, -163.1040649, 139.4189606

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_B1_B2_B1_A1

### Relational analysis result of NS_B2_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514743, upper bound: 18.9511704
time: 0.75 seconds

## Relational analysis of NS_B2_A2_B1_B2_B1_A2

### Relational analysis result of NS_B2_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514743, upper bound: 18.9511704
time: 0.85 seconds

## BFS NS instance: NS_B2_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -47.1688538, 49.4535179, -15.9651003, 17.8168488, -63.9956741, 64.7665710
1: -333.0364685, 114.6190720, -126.3185043, 39.5341644, -361.7195435, 239.9608765
2: -185.1144257, 105.9338226, -65.3370438, 37.7563095, -216.6585388, 170.3313293
3: -232.8503418, 86.1017761, -86.0414810, 31.0704975, -255.6578217, 172.1005402
4: -133.3565216, 92.3568115, -45.5356178, 33.2308121, -161.5912781, 137.4425354

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_B1_B2_B2_A1

### Relational analysis result of NS_B2_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514743, upper bound: 18.9511704
time: 0.75 seconds

## Relational analysis of NS_B2_A2_B1_B2_B2_A2

### Relational analysis result of NS_B2_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514743, upper bound: 18.9511704
time: 0.86 seconds

## BFS NS instance: NS_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -49.2961617, 51.8081856, -53.6824722, 57.0943604, -104.2472076, 103.4547653
1: -348.1985474, 119.8371582, -379.9264526, 130.5429688, -464.5214539, 484.7106018
2: -193.2834930, 111.1613693, -209.8783875, 121.6855392, -305.9817505, 312.0566101
3: -243.3537598, 90.2618637, -264.9808960, 99.1817245, -332.2531738, 344.8234253
4: -139.2274780, 97.2031631, -151.1158447, 107.4557877, -239.5755310, 241.6570282

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B2_A2_B2_A1_B1_B1

### Relational analysis result of NS_B2_A2_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9502016, upper bound: 18.9503553
time: 0.82 seconds

## Relational analysis of NS_B2_A2_B2_A1_B1_B2

### Relational analysis result of NS_B2_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511544, upper bound: 18.9513850
time: 0.76 seconds

## BFS NS instance: NS_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -49.2961617, 51.8081856, -53.2868118, 56.7568626, -103.9019623, 103.1080399
1: -348.1985474, 119.8371582, -377.3058777, 129.5177307, -463.5806885, 481.9603577
2: -193.2834930, 111.1613693, -208.2575378, 120.8274307, -305.0736084, 310.6133728
3: -243.3537598, 90.2618637, -263.0657043, 98.5030746, -331.5576477, 342.9310608
4: -139.2274780, 97.2031631, -149.9602356, 106.8106613, -238.8825226, 240.6682587

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B2_A2_B2_A1_B2_B1

### Relational analysis result of NS_B2_A2_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9502016, upper bound: 18.9503553
time: 0.71 seconds

## Relational analysis of NS_B2_A2_B2_A1_B2_B2

### Relational analysis result of NS_B2_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511544, upper bound: 18.9513850
time: 0.71 seconds

## BFS NS instance: NS_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -48.9295502, 51.5034065, -53.6824722, 57.0943604, -103.9326935, 103.1408539
1: -345.8834839, 118.8759384, -379.9264526, 130.5429688, -462.0512390, 483.8547974
2: -191.7705536, 110.3737793, -209.8783875, 121.6855392, -304.6769714, 311.2265625
3: -241.6047211, 89.6515427, -264.9808960, 99.1817245, -330.5604858, 344.1951294
4: -138.1627808, 96.6145477, -151.1158447, 107.4557877, -238.6847992, 241.0249176

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B2_A2_B2_A2_B1_B1

### Relational analysis result of NS_B2_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9502008, upper bound: 18.9503311
time: 0.91 seconds

## Relational analysis of NS_B2_A2_B2_A2_B1_B2

### Relational analysis result of NS_B2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511528, upper bound: 18.9511528
time: 0.77 seconds

## BFS NS instance: NS_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -48.9295502, 51.5034065, -53.2868118, 56.7568626, -103.6324921, 102.8369827
1: -345.8834839, 118.8759384, -377.3058777, 129.5177307, -461.3757019, 481.3561401
2: -191.7705536, 110.3737793, -208.2575378, 120.8274307, -303.9789124, 309.9822693
3: -241.6047211, 89.6515427, -263.0657043, 98.5030746, -330.0580444, 342.4861145
4: -138.1627808, 96.6145477, -149.9602356, 106.8106613, -238.1361542, 240.1734619

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B2_A2_B2_A2_B2_B1

### Relational analysis result of NS_B2_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9502008, upper bound: 18.9503311
time: 0.78 seconds

## Relational analysis of NS_B2_A2_B2_A2_B2_B2

### Relational analysis result of NS_B2_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511528, upper bound: 18.9511528
time: 1.06 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.18 seconds
NS_B1_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9517439, upper bound: 18.9514176
NS_B1_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9517439, upper bound: 18.9515790
NS_B1_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9517439, upper bound: 18.9514176
NS_B1_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9517439, upper bound: 18.9515790
NS_B1_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9513517, upper bound: 18.9513900
NS_B1_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9516339, upper bound: 18.9515790
NS_B1_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9513517, upper bound: 18.9513900
NS_B1_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9516339, upper bound: 18.9515790
NS_B1_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9512602, upper bound: 18.9514627
NS_B1_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9515720, upper bound: 18.9517601
NS_B1_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9512602, upper bound: 18.9514627
NS_B1_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9515720, upper bound: 18.9517601
NS_B1_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9512602, upper bound: 18.9513843
NS_B1_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9515720, upper bound: 18.9515720
NS_B1_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9512602, upper bound: 18.9513843
NS_B1_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9515720, upper bound: 18.9515720
NS_B1_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9517001, upper bound: 18.9514667
NS_B1_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9517001, upper bound: 18.9514667
NS_B1_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9517001, upper bound: 18.9514578
NS_B1_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9517001, upper bound: 18.9514578
NS_B1_A2_B1_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9506431, upper bound: 18.9502601
NS_B1_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9506431, upper bound: 18.9513819
NS_B1_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9511453, upper bound: 18.9509827
NS_B1_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9511453, upper bound: 18.9511830
NS_B1_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9515087, upper bound: 18.9513244
NS_B1_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9514740, upper bound: 18.9514137
NS_B1_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9515087, upper bound: 18.9513205
NS_B1_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9514723, upper bound: 18.9514092
NS_B1_A2_B2_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9506162, upper bound: 18.9502601
NS_B1_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9512083, upper bound: 18.9513819
NS_B1_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9510912, upper bound: 18.9509827
NS_B1_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9512039, upper bound: 18.9511830
NS_B2_A1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9507970, upper bound: 18.9514850
NS_B2_A1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9507970, upper bound: 18.9514850
NS_B2_A1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9508015, upper bound: 18.9514850
NS_B2_A1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9508015, upper bound: 18.9514850
NS_B2_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9503661, upper bound: 18.9510133
NS_B2_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9504921, upper bound: 18.9511217
NS_B2_A1_A2_B1_B1_B1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9505778, upper bound: 18.9508942
NS_B2_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9507208, upper bound: 18.9509279
NS_B2_A1_A2_B1_B2_B1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9506660, upper bound: 18.9508942
NS_B2_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9507273, upper bound: 18.9509279
NS_B2_A2_B1_B1_B1_A1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9507412, upper bound: 18.9505126
NS_B2_A2_B1_B1_B1_A2, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9507412, upper bound: 18.9505126
NS_B2_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9514743, upper bound: 18.9511704
NS_B2_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9514743, upper bound: 18.9511704
NS_B2_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9514743, upper bound: 18.9511704
NS_B2_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9514743, upper bound: 18.9511704
NS_B2_A2_B2_A1_B1_B1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9502016, upper bound: 18.9503553
NS_B2_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9511544, upper bound: 18.9513850
NS_B2_A2_B2_A1_B2_B1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9502016, upper bound: 18.9503553
NS_B2_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9511544, upper bound: 18.9513850
NS_B2_A2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9502008, upper bound: 18.9503311
NS_B2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9511528, upper bound: 18.9511528
NS_B2_A2_B2_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9502008, upper bound: 18.9503311
NS_B2_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 0, lower bound: -18.9511528, upper bound: 18.9511528

## BFS NS instance: NS_B1_A1_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -9.1916323, 9.8761654, -7.8779206, 8.4777651, -17.6693974, 17.7540836
1: -74.8987808, 22.7326355, -64.7920761, 19.4785786, -94.3773575, 87.5247116
2: -38.4325409, 21.5226784, -33.1352615, 18.4827232, -56.9152641, 54.6579399
3: -50.9241524, 17.4462967, -43.9937706, 14.9916267, -65.9157791, 61.4400673
4: -26.6297913, 18.2700729, -22.8569946, 15.6749372, -42.3047218, 41.1270676

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A1_B1_B1_A1_B1_A1

### Relational analysis result of NS_B1_A1_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517877, upper bound: 18.9517446
time: 0.71 seconds

## Relational analysis of NS_B1_A1_B1_B1_A1_B1_A2

### Relational analysis result of NS_B1_A1_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517788, upper bound: 18.9517443
time: 0.72 seconds

## BFS NS instance: NS_B1_A1_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -9.4717751, 10.2448521, -9.1462526, 9.9101791, -19.3819542, 19.3911018
1: -78.1198044, 23.4605370, -75.9597244, 22.6559219, -100.7757263, 99.4202576
2: -39.8195114, 22.2731819, -38.6012573, 21.5491161, -61.3686295, 60.8744392
3: -53.0109177, 18.0778980, -51.4918709, 17.4865685, -70.4974670, 69.5697708
4: -27.5278988, 18.9055653, -26.6195221, 18.2683506, -45.7962494, 45.5250854

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B1_A1_B1_B1_A1_B2_A1

### Relational analysis result of NS_B1_A1_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517564, upper bound: 18.9518228
time: 0.74 seconds

## Relational analysis of NS_B1_A1_B1_B1_A1_B2_A2

### Relational analysis result of NS_B1_A1_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517564, upper bound: 18.9518228
time: 0.72 seconds

## BFS NS instance: NS_B1_A1_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.6800919, 9.3261433, -7.8779206, 8.4777651, -17.1578560, 17.2040634
1: -70.7416611, 21.4176044, -64.7920761, 19.4785786, -90.2202377, 86.2096786
2: -36.4005013, 20.3121758, -33.1352615, 18.4827232, -54.8832245, 53.4474335
3: -48.0819206, 16.4590206, -43.9937706, 14.9916267, -63.0735435, 60.4527855
4: -25.0920486, 17.2587814, -22.8569946, 15.6749372, -40.7669868, 40.1157722

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

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

## Relational analysis of NS_B1_A1_B1_B1_A2_B1_A1

### Relational analysis result of NS_B1_A1_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517435, upper bound: 18.9514176
time: 0.64 seconds

## Relational analysis of NS_B1_A1_B1_B1_A2_B1_A2

### Relational analysis result of NS_B1_A1_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517435, upper bound: 18.9514138
time: 1.10 seconds

## BFS NS instance: NS_B1_A1_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -8.9127283, 9.6311302, -9.1462526, 9.9101791, -18.8229065, 18.7773819
1: -73.6016846, 22.0697041, -75.9597244, 22.6559219, -96.2575760, 98.0294266
2: -37.6338348, 20.9610996, -38.6012573, 21.5491161, -59.1829529, 59.5623474
3: -49.9398880, 16.9975815, -51.4918709, 17.4865685, -67.4264374, 68.4894562
4: -25.8828659, 17.7806625, -26.6195221, 18.2683506, -44.1512146, 44.4001808

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

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

## Relational analysis of NS_B1_A1_B1_B1_A2_B2_A1

### Relational analysis result of NS_B1_A1_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517435, upper bound: 18.9515790
time: 0.70 seconds

## Relational analysis of NS_B1_A1_B1_B1_A2_B2_A2

### Relational analysis result of NS_B1_A1_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517439, upper bound: 18.9515715
time: 0.78 seconds

## BFS NS instance: NS_B1_A1_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -9.1916323, 9.8761654, -7.6359224, 8.1969728, -17.3886051, 17.5120888
1: -74.8987808, 22.7326355, -62.2804756, 18.8529053, -93.7516861, 85.0130920
2: -38.4325409, 21.5226784, -32.1149025, 17.8938713, -56.3264122, 53.6375809
3: -50.9241524, 17.4462967, -42.3372154, 14.4861479, -65.4103012, 59.7835121
4: -26.6297913, 18.2700729, -22.1040764, 15.1908331, -41.8206215, 40.3741493

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A1_B1_B2_A1_B1_A1

### Relational analysis result of NS_B1_A1_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513428, upper bound: 18.9514845
time: 0.69 seconds

## Relational analysis of NS_B1_A1_B1_B2_A1_B1_A2

### Relational analysis result of NS_B1_A1_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513428, upper bound: 18.9514806
time: 0.79 seconds

## BFS NS instance: NS_B1_A1_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -9.4717751, 10.2448521, -8.4778757, 9.1779327, -18.6497078, 18.7227211
1: -78.1198044, 23.4605370, -70.7042542, 21.0362473, -99.1560516, 94.1647949
2: -39.8195114, 22.2731819, -36.0072632, 20.0005913, -59.8201027, 58.2804451
3: -53.0109177, 18.0778980, -47.9156151, 16.2173748, -69.2282867, 65.9935074
4: -27.5278988, 18.9055653, -24.6969433, 16.9226646, -44.4505577, 43.6025047

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B1_A1_B1_B2_A1_B2_A1

### Relational analysis result of NS_B1_A1_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514877, upper bound: 18.9517960
time: 0.75 seconds

## Relational analysis of NS_B1_A1_B1_B2_A1_B2_A2

### Relational analysis result of NS_B1_A1_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514877, upper bound: 18.9518116
time: 0.76 seconds

## BFS NS instance: NS_B1_A1_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -8.6800919, 9.3261433, -7.6359224, 8.1969728, -16.8770638, 16.9620667
1: -70.7416611, 21.4176044, -62.2804756, 18.8529053, -89.5945663, 83.6980591
2: -36.4005013, 20.3121758, -32.1149025, 17.8938713, -54.2943649, 52.4270782
3: -48.0819206, 16.4590206, -42.3372154, 14.4861479, -62.5680695, 58.7962265
4: -25.0920486, 17.2587814, -22.1040764, 15.1908331, -40.2828827, 39.3628502

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A1_B1_B2_A2_B1_A1

### Relational analysis result of NS_B1_A1_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513428, upper bound: 18.9513900
time: 0.80 seconds

## Relational analysis of NS_B1_A1_B1_B2_A2_B1_A2

### Relational analysis result of NS_B1_A1_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513428, upper bound: 18.9513879
time: 0.67 seconds

## BFS NS instance: NS_B1_A1_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -8.9127283, 9.6311302, -8.4778757, 9.1779327, -18.0906601, 18.1090031
1: -73.6016846, 22.0697041, -70.7042542, 21.0362473, -94.6379089, 92.7739563
2: -37.6338348, 20.9610996, -36.0072632, 20.0005913, -57.6344261, 56.9683571
3: -49.9398880, 16.9975815, -47.9156151, 16.2173748, -66.1572647, 64.9131927
4: -25.8828659, 17.7806625, -24.6969433, 16.9226646, -42.8055229, 42.4776001

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B1_A1_B1_B2_A2_B2_A1

### Relational analysis result of NS_B1_A1_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514258, upper bound: 18.9512657
time: 0.69 seconds

## Relational analysis of NS_B1_A1_B1_B2_A2_B2_A2

### Relational analysis result of NS_B1_A1_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514258, upper bound: 18.9515790
time: 0.76 seconds

## BFS NS instance: NS_B1_A1_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -9.3536158, 10.0446291, -9.8078823, 10.7737017, -20.1273174, 19.8525047
1: -75.9767685, 23.1266975, -81.9743423, 24.3109112, -100.2876740, 105.1010437
2: -39.0535355, 21.8911839, -41.6368141, 23.2403336, -62.2938538, 63.5279922
3: -51.6864891, 17.7446365, -55.4785461, 18.9317894, -70.6182785, 73.2231827
4: -27.0699558, 18.5964527, -28.5661716, 19.8518772, -46.9218330, 47.1626244

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_B2_A1_B1_B1_A1

### Relational analysis result of NS_B1_A1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517878, upper bound: 18.9517572
time: 0.74 seconds

## Relational analysis of NS_B1_A1_B2_A1_B1_B1_A2

### Relational analysis result of NS_B1_A1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517878, upper bound: 18.9517477
time: 0.69 seconds

## BFS NS instance: NS_B1_A1_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -9.6294250, 10.4090176, -10.9999781, 12.1178217, -21.7472458, 21.4089928
1: -79.1715240, 23.8506908, -92.4977417, 27.3242111, -106.4957352, 116.3484192
2: -40.4075851, 22.6325455, -46.7709122, 26.1234493, -66.5310364, 69.4034576
3: -53.7475319, 18.3693523, -62.5494919, 21.2794132, -75.0269470, 80.9188461
4: -27.9642506, 19.2226486, -32.0969505, 22.2817554, -50.2460022, 51.3195992

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_B2_A1_B1_B2_A1

### Relational analysis result of NS_B1_A1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517878, upper bound: 18.9517878
time: 0.72 seconds

## Relational analysis of NS_B1_A1_B2_A1_B1_B2_A2

### Relational analysis result of NS_B1_A1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517878, upper bound: 18.9517948
time: 0.73 seconds

## BFS NS instance: NS_B1_A1_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -9.3536158, 10.0446291, -9.6284409, 10.5404072, -19.8940239, 19.6730690
1: -75.9767685, 23.1266975, -80.1577148, 23.8727131, -99.8494720, 103.2844086
2: -39.0535355, 21.8911839, -40.9374542, 22.7803631, -61.8339005, 62.8286324
3: -51.6864891, 17.7446365, -54.3047714, 18.5335445, -70.2200317, 72.0494080
4: -27.0699558, 18.5964527, -28.0474491, 19.4226131, -46.4925613, 46.6438980

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_B2_A1_B2_B1_A1

### Relational analysis result of NS_B1_A1_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512594, upper bound: 18.9514558
time: 0.67 seconds

## Relational analysis of NS_B1_A1_B2_A1_B2_B1_A2

### Relational analysis result of NS_B1_A1_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512594, upper bound: 18.9514627
time: 0.89 seconds

## BFS NS instance: NS_B1_A1_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -9.6294250, 10.4090176, -10.4251442, 11.4689245, -21.0983505, 20.8341599
1: -79.1715240, 23.8506908, -88.1475067, 25.9286728, -105.1001892, 111.9981766
2: -40.4075851, 22.6325455, -44.6080704, 24.7733231, -65.1809006, 67.2406158
3: -53.7475319, 18.3693523, -59.5870361, 20.1639194, -73.9114380, 77.9563904
4: -27.9642506, 19.2226486, -30.5006104, 21.0625610, -49.0268097, 49.7232590

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_B2_A1_B2_B2_A1

### Relational analysis result of NS_B1_A1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515708, upper bound: 18.9517444
time: 0.78 seconds

## Relational analysis of NS_B1_A1_B2_A1_B2_B2_A2

### Relational analysis result of NS_B1_A1_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515708, upper bound: 18.9517601
time: 0.77 seconds

## BFS NS instance: NS_B1_A1_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -8.9212198, 9.5785580, -9.8078823, 10.7737017, -19.6949215, 19.3864346
1: -72.3731537, 22.0010662, -81.9743423, 24.3109112, -96.6840439, 103.9754105
2: -37.3153839, 20.8486938, -41.6368141, 23.2403336, -60.5557137, 62.4855042
3: -49.2183647, 16.9031143, -55.4785461, 18.9317894, -68.1501541, 72.3816605
4: -25.7608566, 17.7342129, -28.5661716, 19.8518772, -45.6127319, 46.3003845

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_B2_A2_B1_B1_A1

### Relational analysis result of NS_B1_A1_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517439, upper bound: 18.9514137
time: 0.69 seconds

## Relational analysis of NS_B1_A1_B2_A2_B1_B1_A2

### Relational analysis result of NS_B1_A1_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517439, upper bound: 18.9514123
time: 0.83 seconds

## BFS NS instance: NS_B1_A1_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -9.1461706, 9.8753023, -10.9999781, 12.1178217, -21.2639923, 20.8752785
1: -75.1938477, 22.6219559, -92.4977417, 27.3242111, -102.5180435, 115.1196899
2: -38.5208168, 21.4751339, -46.7709122, 26.1234493, -64.6442642, 68.2460403
3: -51.0395317, 17.4175835, -62.5494919, 21.2794132, -72.3189468, 79.9670563
4: -26.5149117, 18.2365627, -32.0969505, 22.2817554, -48.7966690, 50.3335114

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_B2_A2_B1_B2_A1

### Relational analysis result of NS_B1_A1_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517444, upper bound: 18.9515720
time: 0.75 seconds

## Relational analysis of NS_B1_A1_B2_A2_B1_B2_A2

### Relational analysis result of NS_B1_A1_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517444, upper bound: 18.9515708
time: 0.83 seconds

## BFS NS instance: NS_B1_A1_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -8.9212198, 9.5785580, -9.6284409, 10.5404072, -19.4616280, 19.2069988
1: -72.3731537, 22.0010662, -80.1577148, 23.8727131, -96.2458420, 102.1587753
2: -37.3153839, 20.8486938, -40.9374542, 22.7803631, -60.0957489, 61.7861404
3: -49.2183647, 16.9031143, -54.3047714, 18.5335445, -67.7519073, 71.2078857
4: -25.7608566, 17.7342129, -28.0474491, 19.4226131, -45.1834717, 45.7816582

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_B2_A2_B2_B1_A1

### Relational analysis result of NS_B1_A1_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512594, upper bound: 18.9513843
time: 0.67 seconds

## Relational analysis of NS_B1_A1_B2_A2_B2_B1_A2

### Relational analysis result of NS_B1_A1_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512594, upper bound: 18.9513843
time: 0.76 seconds

## BFS NS instance: NS_B1_A1_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -9.1461706, 9.8753023, -10.4251442, 11.4689245, -20.6150951, 20.3004456
1: -75.1938477, 22.6219559, -88.1475067, 25.9286728, -101.1225052, 110.7694473
2: -38.5208168, 21.4751339, -44.6080704, 24.7733231, -63.2941322, 66.0832062
3: -51.0395317, 17.4175835, -59.5870361, 20.1639194, -71.2034531, 77.0046158
4: -26.5149117, 18.2365627, -30.5006104, 21.0625610, -47.5774727, 48.7371750

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_B2_A2_B2_B2_A1

### Relational analysis result of NS_B1_A1_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515708, upper bound: 18.9515720
time: 0.85 seconds

## Relational analysis of NS_B1_A1_B2_A2_B2_B2_A2

### Relational analysis result of NS_B1_A1_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515708, upper bound: 18.9515708
time: 0.72 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -16.7184811, 18.6352940, -9.2110405, 9.9745607, -26.6930370, 27.8463345
1: -131.9722900, 41.4072800, -76.3690186, 22.8215199, -154.7937775, 117.7762909
2: -68.1971436, 39.5120010, -38.8359337, 21.6892929, -89.8864365, 78.3479309
3: -89.8993073, 32.5176506, -51.7815933, 17.6024361, -107.5017395, 84.2992401
4: -47.6973648, 34.7597809, -26.7976322, 18.3903713, -66.0877228, 61.5574112

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B1_B1_A1_A1_A1

### Relational analysis result of NS_B1_A2_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9516795, upper bound: 18.9513244
time: 0.73 seconds

## Relational analysis of NS_B1_A2_B1_B1_A1_A1_A2

### Relational analysis result of NS_B1_A2_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9516548, upper bound: 18.9514137
time: 0.81 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -16.0785961, 17.9471645, -9.2110405, 9.9745607, -26.0531521, 27.1582050
1: -127.4124222, 39.8443184, -76.3690186, 22.8215199, -150.2339478, 116.2133331
2: -65.8345261, 38.0591888, -38.8359337, 21.6892929, -87.5238190, 76.8951111
3: -86.7486877, 31.3225708, -51.7815933, 17.6024361, -104.3511124, 83.1041641
4: -45.8700142, 33.4833336, -26.7976322, 18.3903713, -64.2603760, 60.2809677

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B1_B1_A1_A2_A1

### Relational analysis result of NS_B1_A2_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9516795, upper bound: 18.9513244
time: 0.64 seconds

## Relational analysis of NS_B1_A2_B1_B1_A1_A2_A2

### Relational analysis result of NS_B1_A2_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9516548, upper bound: 18.9514137
time: 0.81 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -16.7693481, 18.6896248, -9.2110405, 9.9745607, -26.7439041, 27.9006653
1: -132.2018280, 41.5270500, -76.3690186, 22.8215199, -155.0233154, 117.8960571
2: -68.3517227, 39.6169701, -38.8359337, 21.6892929, -90.0410156, 78.4529037
3: -90.0680161, 32.6105843, -51.7815933, 17.6024361, -107.6704559, 84.3921814
4: -47.8263245, 34.8677177, -26.7976322, 18.3903713, -66.2166824, 61.6653481

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B1_B1_A2_A1_A1

### Relational analysis result of NS_B1_A2_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9516795, upper bound: 18.9513205
time: 1.13 seconds

## Relational analysis of NS_B1_A2_B1_B1_A2_A1_A2

### Relational analysis result of NS_B1_A2_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9516506, upper bound: 18.9514093
time: 0.68 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -16.1262608, 17.9950829, -9.2110405, 9.9745607, -26.1008205, 27.2061234
1: -127.5614929, 39.9533157, -76.3690186, 22.8215199, -150.3830109, 116.3223343
2: -65.9616241, 38.1489487, -38.8359337, 21.6892929, -87.6509171, 76.9848785
3: -86.8690796, 31.4041710, -51.7815933, 17.6024361, -104.4715118, 83.1857605
4: -45.9843521, 33.5799866, -26.7976322, 18.3903713, -64.3747177, 60.3776131

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B1_B1_A2_A2_A1

### Relational analysis result of NS_B1_A2_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9516795, upper bound: 18.9513205
time: 0.77 seconds

## Relational analysis of NS_B1_A2_B1_B1_A2_A2_A2

### Relational analysis result of NS_B1_A2_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9516506, upper bound: 18.9514092
time: 0.85 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -16.5993710, 18.4320107, -11.0383167, 12.1576433, -28.7570152, 29.4703274
1: -130.5118561, 41.1055984, -92.7541504, 27.4142189, -157.9260712, 133.8597412
2: -67.6486740, 39.1617584, -46.9151840, 26.2082443, -93.8569183, 86.0769348
3: -88.9749069, 32.2032471, -62.7298317, 21.3471699, -110.3220673, 94.9330750
4: -47.3373489, 34.3898697, -32.2015495, 22.3573780, -69.6947250, 66.5914154

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A2_B1_B2_A1_A2_A1

### Relational analysis result of NS_B1_A2_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515896, upper bound: 18.9513819
time: 0.82 seconds

## Relational analysis of NS_B1_A2_B1_B2_A1_A2_A2

### Relational analysis result of NS_B1_A2_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515896, upper bound: 18.9513781
time: 0.81 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -15.5494242, 17.2279491, -10.7254066, 11.7484436, -27.2978668, 27.9533558
1: -119.7535248, 38.3949013, -89.2214279, 26.5579147, -146.3114319, 127.6163177
2: -62.7093582, 36.5225563, -45.3364792, 25.3603172, -88.0696716, 81.8590088
3: -81.8758545, 30.0441360, -60.4401970, 20.6333790, -102.5092163, 90.4843292
4: -44.0536766, 32.2491264, -31.1966877, 21.6419392, -65.6956177, 63.4458160

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A2_B1_B2_A2_A1_A1

### Relational analysis result of NS_B1_A2_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9506431, upper bound: 18.9509827
time: 0.76 seconds

## Relational analysis of NS_B1_A2_B1_B2_A2_A1_A2

### Relational analysis result of NS_B1_A2_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511453, upper bound: 18.9509827
time: 0.71 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -16.4039478, 18.1783867, -11.0030918, 12.1138954, -28.5178432, 29.1814766
1: -127.6782455, 40.5819321, -92.4039230, 27.3215542, -154.9997864, 132.9858551
2: -66.5036621, 38.6202927, -46.7458038, 26.1176567, -92.6213226, 85.3660889
3: -87.1804123, 31.7476692, -62.4984436, 21.2719975, -108.4524078, 94.2461090
4: -46.6688843, 33.9808578, -32.0906105, 22.2799282, -68.9488068, 66.0714569

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A2_B1_B2_A2_A2_A1

### Relational analysis result of NS_B1_A2_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511453, upper bound: 18.9511481
time: 0.81 seconds

## Relational analysis of NS_B1_A2_B1_B2_A2_A2_A2

### Relational analysis result of NS_B1_A2_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514081, upper bound: 18.9511830
time: 0.86 seconds

## BFS NS instance: NS_B1_A2_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -15.8891907, 17.6805553, -8.2767038, 8.9781590, -24.8673496, 25.9572582
1: -126.3404465, 39.4108124, -69.0301514, 20.5370235, -146.8774719, 108.4409561
2: -65.3330536, 37.5636292, -35.1364670, 19.5536270, -84.8866806, 72.7000885
3: -86.0906754, 30.8696899, -46.7553749, 15.8618650, -101.9525223, 77.6250610
4: -45.4898529, 32.8960609, -24.0967426, 16.5752926, -62.0651436, 56.9928055

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_B2_B1_A1_A1_B1

### Relational analysis result of NS_B1_A2_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512468, upper bound: 18.9511123
time: 0.72 seconds

## Relational analysis of NS_B1_A2_B2_B1_A1_A1_B2

### Relational analysis result of NS_B1_A2_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513630, upper bound: 18.9512152
time: 0.70 seconds

## BFS NS instance: NS_B1_A2_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -16.3768330, 18.2392406, -8.5133162, 9.2163868, -25.5932198, 26.7525558
1: -129.1945801, 40.5505714, -70.9173203, 21.1176605, -150.3122253, 111.4678802
2: -66.8744354, 38.6978836, -36.1269989, 20.0783863, -86.9528198, 74.8248672
3: -88.0276947, 31.8302479, -48.0674706, 16.2792053, -104.3069000, 79.8977203
4: -46.6806145, 34.0414238, -24.7886410, 16.9937916, -63.6744080, 58.8300629

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_B2_B1_A1_A2_B1

### Relational analysis result of NS_B1_A2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512265, upper bound: 18.9511518
time: 0.75 seconds

## Relational analysis of NS_B1_A2_B2_B1_A1_A2_B2

### Relational analysis result of NS_B1_A2_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513404, upper bound: 18.9512977
time: 0.75 seconds

## BFS NS instance: NS_B1_A2_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -15.9417048, 17.7344627, -8.2767038, 8.9781590, -24.9198608, 26.0111637
1: -126.5448456, 39.5319366, -69.0301514, 20.5370235, -147.0818634, 108.5620728
2: -65.4853973, 37.6666794, -35.1364670, 19.5536270, -85.0390244, 72.8031387
3: -86.2471313, 30.9614697, -46.7553749, 15.8618650, -102.1089859, 77.7168350
4: -45.6190605, 33.0026398, -24.0967426, 16.5752926, -62.1943474, 57.0993805

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_B2_B1_A2_A1_B1

### Relational analysis result of NS_B1_A2_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512468, upper bound: 18.9511075
time: 0.72 seconds

## Relational analysis of NS_B1_A2_B2_B1_A2_A1_B2

### Relational analysis result of NS_B1_A2_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513630, upper bound: 18.9512124
time: 0.72 seconds

## BFS NS instance: NS_B1_A2_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -16.4295425, 18.2913971, -8.5133162, 9.2163868, -25.6459293, 26.8047123
1: -129.3719635, 40.6699867, -70.9173203, 21.1176605, -150.4895935, 111.5873108
2: -67.0188293, 38.7970467, -36.1269989, 20.0783863, -87.0972137, 74.9240341
3: -88.1681290, 31.9192314, -48.0674706, 16.2792053, -104.4473343, 79.9867020
4: -46.8091812, 34.1457939, -24.7886410, 16.9937916, -63.8029709, 58.9344330

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_B2_B1_A2_A2_B1

### Relational analysis result of NS_B1_A2_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512243, upper bound: 18.9511485
time: 0.77 seconds

## Relational analysis of NS_B1_A2_B2_B1_A2_A2_B2

### Relational analysis result of NS_B1_A2_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513368, upper bound: 18.9512939
time: 0.77 seconds

## BFS NS instance: NS_B1_A2_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -16.4019871, 18.2471237, -10.4773054, 11.5268831, -27.9288635, 28.7244301
1: -129.6527863, 40.6570663, -88.4940643, 26.0537930, -155.7065582, 129.1511230
2: -67.0961304, 38.7575188, -44.7993202, 24.8941517, -91.9902802, 83.5568390
3: -88.3365631, 31.8815079, -59.8270493, 20.2628212, -108.5993805, 91.7085571
4: -46.8088989, 34.0267792, -30.6408958, 21.1746635, -67.9835587, 64.6676788

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A2_B2_B2_A1_A2_A1

### Relational analysis result of NS_B1_A2_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512083, upper bound: 18.9513819
time: 0.83 seconds

## Relational analysis of NS_B1_A2_B2_B2_A1_A2_A2

### Relational analysis result of NS_B1_A2_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512083, upper bound: 18.9513781
time: 0.75 seconds

## BFS NS instance: NS_B1_A2_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -15.4731779, 17.1630802, -10.2083340, 11.1912813, -26.6644535, 27.3714142
1: -119.6978836, 38.2136154, -85.3889236, 25.3456707, -145.0435486, 123.6025238
2: -62.6106720, 36.3823738, -43.4557991, 24.1803837, -86.7910538, 79.8381653
3: -81.8092194, 29.9249229, -57.8135948, 19.6773472, -101.4865646, 87.7385178
4: -43.8746681, 32.1081276, -29.7696495, 20.5967789, -64.4714355, 61.8777771

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A2_B2_B2_A2_A1_A1

### Relational analysis result of NS_B1_A2_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510912, upper bound: 18.9509827
time: 0.71 seconds

## Relational analysis of NS_B1_A2_B2_B2_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B2_B2_A2_A1_A1

### Relational analysis result of NS_B1_A2_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510050, upper bound: 18.9508828
time: 0.74 seconds

## Relational analysis of NS_B1_A2_B2_B2_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_A2_B2_B2_A2_A1_B1

### Relational analysis result of NS_B1_A2_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510093, upper bound: 18.9509647
time: 0.81 seconds

## Relational analysis of NS_B1_A2_B2_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B1_A2_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_B2_B2_A2_A1_A1

### Relational analysis result of NS_B1_A2_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9506162, upper bound: 18.9509827
time: 0.69 seconds

## Relational analysis of NS_B1_A2_B2_B2_A2_A1_A2

### Relational analysis result of NS_B1_A2_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510912, upper bound: 18.9509827
time: 1.09 seconds

## BFS NS instance: NS_B1_A2_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -16.2021770, 17.9894981, -10.4421654, 11.4849043, -27.6870804, 28.4316635
1: -126.8038864, 40.0890045, -88.1511688, 25.9647903, -152.7686615, 128.2401581
2: -65.8844070, 38.1917076, -44.6359062, 24.8071651, -90.6915512, 82.8276138
3: -86.5216141, 31.4103222, -59.6008415, 20.1909542, -106.7125626, 91.0111618
4: -46.1305161, 33.5989380, -30.5335693, 21.1011372, -67.2316513, 64.1325073

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A2_B2_B2_A2_A2_A1

### Relational analysis result of NS_B1_A2_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512039, upper bound: 18.9511481
time: 0.85 seconds

## Relational analysis of NS_B1_A2_B2_B2_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B2_B2_A2_A2_A1

### Relational analysis result of NS_B1_A2_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511332, upper bound: 18.9511105
time: 0.75 seconds

## Relational analysis of NS_B1_A2_B2_B2_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B1_A2_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B1_A2_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_A2_B2_B2_A2_A2_B1

### Relational analysis result of NS_B1_A2_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510486, upper bound: 18.9511530
time: 0.85 seconds

## Relational analysis of NS_B1_A2_B2_B2_A2_A2_B2

### Relational analysis result of NS_B1_A2_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510486, upper bound: 18.9511830
time: 0.74 seconds

## BFS NS instance: NS_B2_A1_A1_B1_B1_B1

### Backsubstitution after applying NS history:
0: -10.2026196, 11.0001450, -16.7239380, 18.6511040, -28.8537235, 27.7240810
1: -81.1365585, 25.4654350, -132.2078094, 41.4600945, -122.5966492, 157.6732483
2: -42.2413712, 23.9027443, -68.2754517, 39.5594101, -81.8007431, 92.1781921
3: -55.5976448, 19.5095367, -90.0512924, 32.5583267, -88.1559753, 109.5608292
4: -29.5800400, 20.4724350, -47.7286110, 34.7851067, -64.3651428, 68.2010498

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_A1_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_A1_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A1_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B2_A1_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_A1_A1_B1_B1_B1_A1

### Relational analysis result of NS_B2_A1_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9507970, upper bound: 18.9514850
time: 0.73 seconds

## Relational analysis of NS_B2_A1_A1_B1_B1_B1_A2

### Relational analysis result of NS_B2_A1_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9507970, upper bound: 18.9514850
time: 0.84 seconds

## BFS NS instance: NS_B2_A1_A1_B1_B1_B2

### Backsubstitution after applying NS history:
0: -10.2026196, 11.0001450, -16.1620369, 18.0303898, -28.2330093, 27.1621819
1: -81.1365585, 25.4654350, -128.0772858, 40.0749207, -121.2114792, 153.5427246
2: -42.2413712, 23.9027443, -66.1815338, 38.2526588, -80.4940033, 90.0842743
3: -55.5976448, 19.5095367, -87.2107086, 31.4890099, -87.0866547, 106.7202454
4: -29.5800400, 20.4724350, -46.1225510, 33.6365585, -63.2165985, 66.5949860

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_A1_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_A1_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A1_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B2_A1_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_A1_B1_B1_B2_B1

### Relational analysis result of NS_B2_A1_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9506445, upper bound: 18.9512151
time: 0.71 seconds

## Relational analysis of NS_B2_A1_A1_B1_B1_B2_B2

### Relational analysis result of NS_B2_A1_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9506634, upper bound: 18.9512420
time: 0.73 seconds

## BFS NS instance: NS_B2_A1_A1_B1_B2_B1

### Backsubstitution after applying NS history:
0: -10.2026196, 11.0001450, -16.7788544, 18.7100029, -28.9126225, 27.7789955
1: -81.1365585, 25.4654350, -132.4695282, 41.5903320, -122.7268906, 157.9349670
2: -42.2413712, 23.9027443, -68.4477386, 39.6740913, -81.9154510, 92.3504791
3: -55.5976448, 19.5095367, -90.2427902, 32.6592255, -88.2568665, 109.7523270
4: -29.5800400, 20.4724350, -47.8704529, 34.9013100, -64.4813461, 68.3428879

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_A1_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_A1_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_A1_B1_B2_B1_B1

### Relational analysis result of NS_B2_A1_A1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9507418, upper bound: 18.9514258
time: 0.99 seconds

## Relational analysis of NS_B2_A1_A1_B1_B2_B1_B2

### Relational analysis result of NS_B2_A1_A1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9507543, upper bound: 18.9514438
time: 0.82 seconds

## BFS NS instance: NS_B2_A1_A1_B1_B2_B2

### Backsubstitution after applying NS history:
0: -10.2026196, 11.0001450, -16.2079334, 18.0785122, -28.2811317, 27.2080784
1: -81.1365585, 25.4654350, -128.2495880, 40.1809311, -121.3174896, 153.7150116
2: -42.2413712, 23.9027443, -66.3091354, 38.3419266, -80.5832672, 90.2118759
3: -55.5976448, 19.5095367, -87.3408813, 31.5713081, -87.1689529, 106.8504181
4: -29.5800400, 20.4724350, -46.2344131, 33.7324562, -63.3124962, 66.7068481

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_A1_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_A1_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A1_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B2_A1_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_A1_B1_B2_B2_B1

### Relational analysis result of NS_B2_A1_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9506474, upper bound: 18.9512151
time: 0.81 seconds

## Relational analysis of NS_B2_A1_A1_B1_B2_B2_B2

### Relational analysis result of NS_B2_A1_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9506661, upper bound: 18.9512420
time: 0.73 seconds

## BFS NS instance: NS_B2_A1_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -10.1318464, 10.9177208, -16.5022221, 18.2831688, -28.4150162, 27.4199371
1: -80.5541840, 25.2932720, -128.2888336, 40.8149490, -121.3691330, 153.5821075
2: -41.9504547, 23.7347755, -66.8558197, 38.8402443, -80.7906952, 90.5905762
3: -55.2103462, 19.3708324, -87.6132965, 31.9265194, -87.1368637, 106.9841156
4: -29.3772621, 20.3219624, -46.9309120, 34.1891975, -63.5664520, 67.2528763

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_A1_B2_B2_B1_B1

### Relational analysis result of NS_B2_A1_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9503661, upper bound: 18.9510133
time: 0.72 seconds

## Relational analysis of NS_B2_A1_A1_B2_B2_B1_B2

### Relational analysis result of NS_B2_A1_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9503661, upper bound: 18.9510133
time: 0.69 seconds

## BFS NS instance: NS_B2_A1_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -10.1318464, 10.9177208, -16.5390244, 18.3244286, -28.4562759, 27.4567375
1: -80.5541840, 25.2932720, -128.4023743, 40.9025803, -121.4567642, 153.6956482
2: -41.9504547, 23.7347755, -66.9511490, 38.9122810, -80.8627319, 90.6859283
3: -55.2103462, 19.3708324, -87.7039871, 31.9963760, -87.2067261, 107.0748215
4: -29.3772621, 20.3219624, -47.0151749, 34.2725792, -63.6498375, 67.3371353

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_A1_B2_B2_B2_B1

### Relational analysis result of NS_B2_A1_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9504921, upper bound: 18.9511217
time: 0.79 seconds

## Relational analysis of NS_B2_A1_A1_B2_B2_B2_B2

### Relational analysis result of NS_B2_A1_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9504921, upper bound: 18.9511217
time: 0.81 seconds

## BFS NS instance: NS_B2_A1_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -9.6194334, 10.4234314, -16.3826389, 18.2007446, -27.8201733, 26.8060703
1: -77.0738144, 23.9353199, -128.6218109, 40.5453606, -117.6191711, 152.5571289
2: -39.8988647, 22.5433407, -66.7216110, 38.6460304, -78.5448914, 89.2649536
3: -52.6164970, 18.4350414, -87.6923828, 31.7756290, -84.3921204, 106.1274185
4: -27.8285465, 19.3550510, -46.6414719, 33.9838753, -61.8124237, 65.9965134

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B2_A1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_A1_A2_B1_B1_B2_A1

### Relational analysis result of NS_B2_A1_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9507208, upper bound: 18.9509279
time: 0.72 seconds

## Relational analysis of NS_B2_A1_A2_B1_B1_B2_A2

### Relational analysis result of NS_B2_A1_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9507208, upper bound: 18.9509279
time: 0.80 seconds

## BFS NS instance: NS_B2_A1_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -9.6194334, 10.4234314, -16.4303036, 18.2507935, -27.8702221, 26.8537312
1: -77.0738144, 23.9353199, -128.7922363, 40.6554871, -117.7293015, 152.7275543
2: -39.8988647, 22.5433407, -66.8545609, 38.7388763, -78.6377411, 89.3979034
3: -52.6164970, 18.4350414, -87.8238144, 31.8606644, -84.4771500, 106.2588577
4: -27.8285465, 19.3550510, -46.7585335, 34.0841179, -61.9126663, 66.1135712

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B2_A1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_A1_A2_B1_B2_B2_A1

### Relational analysis result of NS_B2_A1_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9507273, upper bound: 18.9509279
time: 0.78 seconds

## Relational analysis of NS_B2_A1_A2_B1_B2_B2_A2

### Relational analysis result of NS_B2_A1_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9507273, upper bound: 18.9509279
time: 0.80 seconds

## BFS NS instance: NS_B2_A2_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -46.9516640, 49.1886787, -16.5459156, 18.4468384, -64.3387146, 65.0727768
1: -331.6154480, 114.1273575, -130.5139160, 40.9471626, -361.6687012, 243.7411346
2: -184.4019775, 105.4152298, -67.4658813, 39.0936546, -217.0114441, 172.0542908
3: -231.9599457, 85.6870499, -88.9198608, 32.1694412, -255.5787659, 174.6069031
4: -132.7855072, 91.8339005, -47.1886597, 34.4116898, -161.9862671, 138.6297455

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_A2_B1_B2_B1_A1_A1

### Relational analysis result of NS_B2_A2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514850, upper bound: 18.9507628
time: 0.76 seconds

## Relational analysis of NS_B2_A2_B1_B2_B1_A1_A2

### Relational analysis result of NS_B2_A2_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511217, upper bound: 18.9504921
time: 0.74 seconds

## BFS NS instance: NS_B2_A2_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -46.6914444, 49.0093880, -16.5459156, 18.4468384, -64.1314468, 64.8774872
1: -330.2214355, 113.4124603, -130.5139160, 40.9471626, -360.0162354, 243.1321106
2: -183.3190765, 104.8923111, -67.4658813, 39.0936546, -216.1275177, 171.4661407
3: -230.8128815, 85.2817383, -88.9198608, 32.1694412, -254.4467621, 174.2015991
4: -132.0337677, 91.4908295, -47.1886597, 34.4116898, -161.4030762, 138.2225189

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_B1_B2_B1_A2_B1

### Relational analysis result of NS_B2_A2_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9516837, upper bound: 18.9511704
time: 0.83 seconds

## Relational analysis of NS_B2_A2_B1_B2_B1_A2_B2

### Relational analysis result of NS_B2_A2_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9516867, upper bound: 18.9511704
time: 0.74 seconds

## BFS NS instance: NS_B2_A2_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -46.8575211, 49.0793800, -15.9651003, 17.8168488, -63.6017723, 64.3679581
1: -330.9858093, 113.9012909, -126.3185043, 39.5341644, -359.5343933, 239.0111237
2: -184.0623779, 105.1884689, -65.3370438, 37.7563095, -215.2711334, 169.4980164
3: -231.5264435, 85.5025024, -86.0414810, 31.0704975, -254.0126343, 171.4078369
4: -132.5339203, 91.6166077, -45.5356178, 33.2308121, -160.5274811, 136.6762085

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_A2_B1_B2_B2_A1_A1

### Relational analysis result of NS_B2_A2_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9509665, upper bound: 18.9507611
time: 0.74 seconds

## Relational analysis of NS_B2_A2_B1_B2_B2_A1_A2

### Relational analysis result of NS_B2_A2_B1_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9508919, upper bound: 18.9504921
time: 0.82 seconds

## BFS NS instance: NS_B2_A2_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -46.6081505, 48.9136086, -15.9651003, 17.8168488, -63.4471054, 64.2170258
1: -329.6679993, 113.2126999, -126.3185043, 39.5341644, -358.1928406, 238.5983429
2: -183.0183563, 104.6920242, -65.3370438, 37.7563095, -214.6217499, 169.0668335
3: -230.4315491, 85.1196442, -86.0414810, 31.0704975, -253.1136627, 171.1080475
4: -131.8106842, 91.2992630, -45.5356178, 33.2308121, -160.1087189, 136.3762207

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_B1_B2_B2_A2_B1

### Relational analysis result of NS_B2_A2_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514743, upper bound: 18.9511704
time: 0.94 seconds

## Relational analysis of NS_B2_A2_B1_B2_B2_A2_B2

### Relational analysis result of NS_B2_A2_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514679, upper bound: 18.9511704
time: 0.78 seconds

## BFS NS instance: NS_B2_A2_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -49.2545700, 51.7601738, -53.5959854, 56.9929428, -104.0943298, 103.3058777
1: -347.8593445, 119.7359772, -379.1450500, 130.3323822, -463.9507751, 483.8278809
2: -193.1196594, 111.0582886, -209.5260773, 121.4646454, -305.5796509, 311.5562744
3: -243.1332855, 90.1819305, -264.4847717, 99.0133133, -331.8513794, 344.2352600
4: -139.1091461, 97.1143799, -150.8627777, 107.2724075, -239.2382965, 241.2522125

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_B2_A1_B1_B2_B1

### Relational analysis result of NS_B2_A2_B2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514225, upper bound: 18.9514932
time: 0.85 seconds

## Relational analysis of NS_B2_A2_B2_A1_B1_B2_B2

### Relational analysis result of NS_B2_A2_B2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514933, upper bound: 18.9514932
time: 0.94 seconds

## BFS NS instance: NS_B2_A2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -49.2545700, 51.7601738, -53.1687469, 56.6217270, -103.7130432, 102.9272766
1: -347.8593445, 119.7359772, -376.4652710, 129.2285767, -462.9304810, 480.9613342
2: -193.1196594, 111.0582886, -207.8087158, 120.5509644, -304.5971985, 310.0133057
3: -243.1332855, 90.1819305, -262.4909058, 98.2759705, -331.0929565, 342.2576904
4: -139.1091461, 97.1143799, -149.6345978, 106.5604401, -238.4715424, 240.1877289

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_B2_A1_B2_B2_B1

### Relational analysis result of NS_B2_A2_B2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9502016, upper bound: 18.9513850
time: 0.82 seconds

## Relational analysis of NS_B2_A2_B2_A1_B2_B2_B2

### Relational analysis result of NS_B2_A2_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9502016, upper bound: 18.9513850
time: 0.82 seconds

## BFS NS instance: NS_B2_A2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -48.8744926, 51.4406471, -53.5959854, 56.9929428, -103.7663727, 102.9760513
1: -345.5014648, 118.7423782, -379.1450500, 130.3323822, -461.4194946, 482.9389343
2: -191.5642090, 110.2456970, -209.5260773, 121.4646454, -304.2304688, 310.6946716
3: -241.3430481, 89.5464325, -264.4847717, 99.0133133, -330.1166687, 343.5804138
4: -138.0116882, 96.4977264, -150.8627777, 107.2724075, -238.3134155, 240.5895691

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_B2_A2_B1_B2_B1

### Relational analysis result of NS_B2_A2_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513440, upper bound: 18.9511544
time: 0.80 seconds

## Relational analysis of NS_B2_A2_B2_A2_B1_B2_B2

### Relational analysis result of NS_B2_A2_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513851, upper bound: 18.9511544
time: 1.69 seconds

## BFS NS instance: NS_B2_A2_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -48.8744926, 51.4406471, -53.1687469, 56.6217270, -103.4300613, 102.6402740
1: -345.5014648, 118.7423782, -376.4652710, 129.2285767, -460.6643372, 480.3234863
2: -191.5642090, 110.2456970, -207.8087158, 120.5509644, -303.4580383, 309.3506165
3: -241.3430481, 89.5464325, -262.4909058, 98.2759705, -329.5512390, 341.7859497
4: -138.0116882, 96.4977264, -149.6345978, 106.5604401, -237.6909637, 239.6624451

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_B2_A2_B2_B2_B1

### Relational analysis result of NS_B2_A2_B2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511520, upper bound: 18.9511528
time: 0.77 seconds

## Relational analysis of NS_B2_A2_B2_A2_B2_B2_B2

### Relational analysis result of NS_B2_A2_B2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9502008, upper bound: 18.9511528
time: 0.77 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.60 seconds
NS_B1_A1_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9517877, upper bound: 18.9517446
NS_B1_A1_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9517788, upper bound: 18.9517443
NS_B1_A1_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9517564, upper bound: 18.9518228
NS_B1_A1_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9517564, upper bound: 18.9518228
NS_B1_A1_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9517435, upper bound: 18.9514176
NS_B1_A1_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9517435, upper bound: 18.9514138
NS_B1_A1_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9517435, upper bound: 18.9515790
NS_B1_A1_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9517439, upper bound: 18.9515715
NS_B1_A1_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9513428, upper bound: 18.9514845
NS_B1_A1_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9513428, upper bound: 18.9514806
NS_B1_A1_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9514877, upper bound: 18.9517960
NS_B1_A1_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9514877, upper bound: 18.9518116
NS_B1_A1_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9513428, upper bound: 18.9513900
NS_B1_A1_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9513428, upper bound: 18.9513879
NS_B1_A1_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9514258, upper bound: 18.9512657
NS_B1_A1_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9514258, upper bound: 18.9515790
NS_B1_A1_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9517878, upper bound: 18.9517572
NS_B1_A1_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9517878, upper bound: 18.9517477
NS_B1_A1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9517878, upper bound: 18.9517878
NS_B1_A1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9517878, upper bound: 18.9517948
NS_B1_A1_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9512594, upper bound: 18.9514558
NS_B1_A1_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9512594, upper bound: 18.9514627
NS_B1_A1_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9515708, upper bound: 18.9517444
NS_B1_A1_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9515708, upper bound: 18.9517601
NS_B1_A1_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9517439, upper bound: 18.9514137
NS_B1_A1_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9517439, upper bound: 18.9514123
NS_B1_A1_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9517444, upper bound: 18.9515720
NS_B1_A1_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9517444, upper bound: 18.9515708
NS_B1_A1_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9512594, upper bound: 18.9513843
NS_B1_A1_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9512594, upper bound: 18.9513843
NS_B1_A1_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9515708, upper bound: 18.9515720
NS_B1_A1_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9515708, upper bound: 18.9515708
NS_B1_A2_B1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9516795, upper bound: 18.9513244
NS_B1_A2_B1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9516548, upper bound: 18.9514137
NS_B1_A2_B1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9516795, upper bound: 18.9513244
NS_B1_A2_B1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9516548, upper bound: 18.9514137
NS_B1_A2_B1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9516795, upper bound: 18.9513205
NS_B1_A2_B1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9516506, upper bound: 18.9514093
NS_B1_A2_B1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9516795, upper bound: 18.9513205
NS_B1_A2_B1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9516506, upper bound: 18.9514092
NS_B1_A2_B1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9515896, upper bound: 18.9513819
NS_B1_A2_B1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9515896, upper bound: 18.9513781
NS_B1_A2_B1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9506431, upper bound: 18.9509827
NS_B1_A2_B1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9511453, upper bound: 18.9509827
NS_B1_A2_B1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9511453, upper bound: 18.9511481
NS_B1_A2_B1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9514081, upper bound: 18.9511830
NS_B1_A2_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9512468, upper bound: 18.9511123
NS_B1_A2_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9513630, upper bound: 18.9512152
NS_B1_A2_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9512265, upper bound: 18.9511518
NS_B1_A2_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9513404, upper bound: 18.9512977
NS_B1_A2_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9512468, upper bound: 18.9511075
NS_B1_A2_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9513630, upper bound: 18.9512124
NS_B1_A2_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9512243, upper bound: 18.9511485
NS_B1_A2_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9513368, upper bound: 18.9512939
NS_B1_A2_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9512083, upper bound: 18.9513819
NS_B1_A2_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9512083, upper bound: 18.9513781
NS_B1_A2_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9506162, upper bound: 18.9509827
NS_B1_A2_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9510912, upper bound: 18.9509827
NS_B1_A2_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9510486, upper bound: 18.9511530
NS_B1_A2_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9510486, upper bound: 18.9511830
NS_B2_A1_A1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9507970, upper bound: 18.9514850
NS_B2_A1_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9507970, upper bound: 18.9514850
NS_B2_A1_A1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9506445, upper bound: 18.9512151
NS_B2_A1_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9506634, upper bound: 18.9512420
NS_B2_A1_A1_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9507418, upper bound: 18.9514258
NS_B2_A1_A1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9507543, upper bound: 18.9514438
NS_B2_A1_A1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9506474, upper bound: 18.9512151
NS_B2_A1_A1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9506661, upper bound: 18.9512420
NS_B2_A1_A1_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9503661, upper bound: 18.9510133
NS_B2_A1_A1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9503661, upper bound: 18.9510133
NS_B2_A1_A1_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9504921, upper bound: 18.9511217
NS_B2_A1_A1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9504921, upper bound: 18.9511217
NS_B2_A1_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9507208, upper bound: 18.9509279
NS_B2_A1_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9507208, upper bound: 18.9509279
NS_B2_A1_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9507273, upper bound: 18.9509279
NS_B2_A1_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9507273, upper bound: 18.9509279
NS_B2_A2_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9514850, upper bound: 18.9507628
NS_B2_A2_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9511217, upper bound: 18.9504921
NS_B2_A2_B1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9516837, upper bound: 18.9511704
NS_B2_A2_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9516867, upper bound: 18.9511704
NS_B2_A2_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9509665, upper bound: 18.9507611
NS_B2_A2_B1_B2_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9508919, upper bound: 18.9504921
NS_B2_A2_B1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9514743, upper bound: 18.9511704
NS_B2_A2_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9514679, upper bound: 18.9511704
NS_B2_A2_B2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9514225, upper bound: 18.9514932
NS_B2_A2_B2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9514933, upper bound: 18.9514932
NS_B2_A2_B2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9502016, upper bound: 18.9513850
NS_B2_A2_B2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9502016, upper bound: 18.9513850
NS_B2_A2_B2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9513440, upper bound: 18.9511544
NS_B2_A2_B2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9513851, upper bound: 18.9511544
NS_B2_A2_B2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9511520, upper bound: 18.9511528
NS_B2_A2_B2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -18.9502008, upper bound: 18.9511528

## BFS NS instance: NS_B1_A1_B1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -9.0949574, 9.7653456, -7.8779206, 8.4777651, -17.5727215, 17.6432629
1: -74.1953583, 22.4964275, -64.7920761, 19.4785786, -93.6739349, 87.2885056
2: -38.0611267, 21.2918739, -33.1352615, 18.4827232, -56.5438499, 54.4271240
3: -50.4441833, 17.2572269, -43.9937706, 14.9916267, -65.4358063, 61.2509956
4: -26.3631191, 18.0579281, -22.8569946, 15.6749372, -42.0380554, 40.9149208

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_B1_B1_A1_B1_A1_A1

### Relational analysis result of NS_B1_A1_B1_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517877, upper bound: 18.9517446
time: 0.75 seconds

## Relational analysis of NS_B1_A1_B1_B1_A1_B1_A1_A2

### Relational analysis result of NS_B1_A1_B1_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517877, upper bound: 18.9517446
time: 0.77 seconds

## BFS NS instance: NS_B1_A1_B1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -9.1824007, 9.8590603, -7.8779206, 8.4777651, -17.6601658, 17.7369804
1: -74.6439209, 22.7037506, -64.7920761, 19.4785786, -94.1224976, 87.4958191
2: -38.3604012, 21.4777966, -33.1352615, 18.4827232, -56.8431244, 54.6130562
3: -50.7735329, 17.4147167, -43.9937706, 14.9916267, -65.7651596, 61.4084854
4: -26.5944252, 18.2403851, -22.8569946, 15.6749372, -42.2693634, 41.0973778

Time for backsubstitution: 1.84 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.63 + 418.05 = 421.68 seconds
