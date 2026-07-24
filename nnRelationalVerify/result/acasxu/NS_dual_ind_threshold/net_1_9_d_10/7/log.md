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
execution time: IAR + RelationalAnalysis = 1.21 + 1.94 = 3.15 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -18.9524357, upper bound: 18.9524357

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9519298, upper bound: 18.9518763
time: 0.77 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9518096, upper bound: 18.9518763
time: 0.72 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.60 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.60
Output dim: 0, lower bound: -18.9519298, upper bound: 18.9518763
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.60
Output dim: 0, lower bound: -18.9518096, upper bound: 18.9518763

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -10.3281603, 11.0515471, -11.0580759, 11.8221369, -22.1502972, 22.1096230
1: -82.6208801, 25.4569778, -87.6119919, 27.2738838, -109.8947525, 113.0689697
2: -42.8500671, 24.0520840, -45.6439209, 25.7299995, -68.5800629, 69.6960068
3: -56.3412666, 19.4773655, -59.8711472, 20.8383999, -77.1796646, 79.3485031
4: -29.7730980, 20.4836941, -31.7946434, 21.9526691, -51.7257690, 52.2783356

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9518096, upper bound: 18.9518096
time: 1.43 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9518096, upper bound: 18.9518763
time: 1.31 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -17.4140701, 19.2958088, -11.3669538, 12.1905842, -29.6046543, 30.6627579
1: -136.1413116, 43.0780411, -88.8660049, 28.2998371, -164.4411469, 131.9440308
2: -70.8038712, 41.0104980, -46.7948532, 26.4937973, -97.2976685, 87.8053513
3: -92.8945236, 33.6828041, -61.1560364, 21.5206909, -114.4152145, 94.8388367
4: -49.5935059, 36.0196648, -32.8203812, 22.6683731, -72.2618713, 68.8400421

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9518452, upper bound: 18.9515270
time: 0.82 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515056, upper bound: 18.9515056
time: 0.88 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.88 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.88
Output dim: 0, lower bound: -18.9518096, upper bound: 18.9518096
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.88
Output dim: 0, lower bound: -18.9518096, upper bound: 18.9518763
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.88
Output dim: 0, lower bound: -18.9518452, upper bound: 18.9515270
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.88
Output dim: 0, lower bound: -18.9515056, upper bound: 18.9515056

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -10.3281603, 11.0515471, -10.3281603, 11.0515471, -21.3797073, 21.3797073
1: -82.6208801, 25.4569778, -82.6208801, 25.4569778, -108.0778580, 108.0778580
2: -42.8500671, 24.0520840, -42.8500671, 24.0520840, -66.9021530, 66.9021530
3: -56.3412666, 19.4773655, -56.3412666, 19.4773655, -75.8186340, 75.8186340
4: -29.7730980, 20.4836941, -29.7730980, 20.4836941, -50.2567902, 50.2567902

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517464, upper bound: 18.9517490
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9518387, upper bound: 18.9517852
time: 0.84 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -10.3281603, 11.0515471, -17.4140701, 19.2958088, -29.6239681, 28.4656162
1: -82.6208801, 25.4569778, -136.1413116, 43.0780411, -125.6989212, 161.5982971
2: -42.8500671, 24.0520840, -70.8038712, 41.0104980, -83.8605652, 94.8559494
3: -56.3412666, 19.4773655, -92.8945236, 33.6828041, -90.0240707, 112.3718872
4: -29.7730980, 20.4836941, -49.5935059, 36.0196648, -65.7927551, 70.0771942

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517464, upper bound: 18.9517941
time: 1.02 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9518387, upper bound: 18.9517852
time: 0.83 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -17.4140701, 19.2958088, -10.6524124, 11.4190826, -28.8331490, 29.9482212
1: -136.1413116, 43.0780411, -83.1773911, 26.5013103, -162.6426239, 126.2554321
2: -70.8038712, 41.0104980, -43.7759018, 24.8135796, -95.6174469, 84.7863998
3: -92.8945236, 33.6828041, -57.1881180, 20.2161560, -113.1106796, 90.8709259
4: -49.5935059, 36.0196648, -30.7249832, 21.2848549, -70.8783493, 66.7446442

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515056, upper bound: 18.9515056
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515056, upper bound: 18.9515056
time: 0.71 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -15.7999792, 17.4130554, -48.3735580, 50.7344971, -65.9435349, 64.8651733
1: -122.4180832, 38.9756927, -341.1856384, 117.5696945, -239.9877625, 369.7427673
2: -63.8048401, 37.0671844, -189.7698975, 108.7521591, -172.2580109, 220.8140564
3: -83.5949783, 30.3982105, -238.6354523, 88.3565750, -171.9515533, 260.9320374
4: -44.8380966, 32.5400124, -136.6828156, 94.8989105, -139.6083832, 164.3392792

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511544, upper bound: 18.9513851
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511528, upper bound: 18.9511528
time: 0.78 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.04 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 0, lower bound: -18.9517464, upper bound: 18.9517490
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 0, lower bound: -18.9518387, upper bound: 18.9517852
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 0, lower bound: -18.9517464, upper bound: 18.9517941
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 0, lower bound: -18.9518387, upper bound: 18.9517852
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 0, lower bound: -18.9515056, upper bound: 18.9515056
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 0, lower bound: -18.9515056, upper bound: 18.9515056
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 0, lower bound: -18.9511544, upper bound: 18.9513851
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.04
Output dim: 0, lower bound: -18.9511528, upper bound: 18.9511528

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -9.5410004, 10.2219028, -9.9075117, 10.6043987, -20.1453991, 20.1294136
1: -77.3461533, 23.5448418, -79.7855530, 24.4337444, -101.7798996, 103.3303986
2: -39.8698807, 22.2895355, -41.2437630, 23.1033478, -62.9732285, 63.5332947
3: -52.6435242, 18.0404015, -54.3544998, 18.7060280, -71.3495483, 72.3948975
4: -27.6078606, 18.9090958, -28.6164150, 19.6349144, -47.2427750, 47.5255051

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9518128, upper bound: 18.9516631
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515896, upper bound: 18.9516413
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -11.3833199, 12.4197531, -10.1348324, 10.8555441, -22.2388649, 22.5545845
1: -94.1245575, 28.1889305, -81.3478851, 24.9925842, -119.1171265, 109.5368042
2: -48.1099281, 26.8477879, -42.1167336, 23.6256142, -71.7355423, 68.9644852
3: -63.8559532, 21.8105049, -55.4342842, 19.1376820, -82.9936371, 77.2447891
4: -33.0945320, 22.8723297, -29.2381802, 20.1096287, -53.2041588, 52.1105080

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517610, upper bound: 18.9515778
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515778, upper bound: 18.9515778
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -9.5410004, 10.2219028, -17.0510025, 18.9151249, -28.4561234, 27.2729053
1: -77.3461533, 23.5448418, -133.6593781, 42.1793594, -119.5254974, 157.2042236
2: -39.8698807, 22.2895355, -69.4119186, 40.1873589, -80.0572357, 91.7014389
3: -52.6435242, 18.0404015, -91.1684952, 33.0156631, -85.6591797, 109.2088852
4: -27.6078606, 18.9090958, -48.5767021, 35.3066254, -62.9144859, 67.4857941

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517459, upper bound: 18.9515634
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514723, upper bound: 18.9515478
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -11.3833199, 12.4197531, -17.2105026, 19.0714111, -30.4547291, 29.6302547
1: -94.1245575, 28.1889305, -134.7238770, 42.5802879, -136.7048492, 162.9128113
2: -48.1099281, 26.8477879, -70.0363464, 40.5367737, -88.6466980, 96.8841171
3: -63.8559532, 21.8105049, -91.9139709, 33.2989769, -97.1549301, 113.7244720
4: -33.0945320, 22.8723297, -49.0324707, 35.5876884, -68.6822128, 71.9048004

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517531, upper bound: 18.9515647
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515447, upper bound: 18.9515514
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -16.9493484, 18.7958755, -10.6524124, 11.4190826, -28.3684254, 29.4482880
1: -132.4742889, 41.8788414, -83.1773911, 26.5013103, -158.9755859, 125.0562286
2: -68.8888321, 39.9076653, -43.7759018, 24.8135796, -93.7024078, 83.6835632
3: -90.4129486, 32.7751465, -57.1881180, 20.2161560, -110.6291046, 89.9632645
4: -48.2513695, 35.0908470, -30.7249832, 21.2848549, -69.5362091, 65.8158264

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9516867, upper bound: 18.9511704
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514743, upper bound: 18.9511704
time: 1.31 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -49.6960869, 52.7572670, -10.6524124, 11.4190826, -60.5189857, 62.9853897
1: -348.9709473, 120.6243744, -83.1773911, 26.5013103, -366.0033264, 203.8017578
2: -193.4904022, 112.2382355, -43.7759018, 24.8135796, -213.4867096, 156.0141144
3: -243.6257935, 91.5296783, -57.1881180, 20.2161560, -256.8230286, 148.7178040
4: -139.6913452, 99.1593933, -30.7249832, 21.2848549, -157.0089111, 129.8843689

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9516867, upper bound: 18.9511704
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514743, upper bound: 18.9511704
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -15.4601688, 17.1181679, -47.9224510, 50.2440910, -65.0811234, 64.0850601
1: -120.7814865, 38.1829414, -338.0102844, 116.4712067, -237.2526855, 365.5260010
2: -62.5406075, 36.3876457, -188.0288239, 107.6844559, -169.8307495, 218.2251740
3: -82.3589401, 29.8769817, -236.4368286, 87.5081406, -169.8670807, 258.0308228
4: -43.9425278, 31.9672260, -135.4282379, 93.9446335, -137.6923828, 162.3900146

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511528, upper bound: 18.9511528
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511528, upper bound: 18.9511528
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -14.9710941, 16.5930767, -47.7686501, 50.1225319, -64.4714890, 63.4386368
1: -117.1259918, 36.9314232, -337.2016602, 116.0738144, -233.0905609, 363.5325928
2: -60.6932373, 35.2148247, -187.3680267, 107.3766098, -167.5891876, 216.5719910
3: -79.8433533, 28.9404850, -235.7194519, 87.2581940, -167.1015472, 256.5898743
4: -42.5029488, 30.9753571, -134.9969482, 93.7052002, -135.9813995, 161.1072540

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511528, upper bound: 18.9511528
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511528, upper bound: 18.9511528
time: 0.79 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.03 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 0, lower bound: -18.9518128, upper bound: 18.9516631
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 0, lower bound: -18.9515896, upper bound: 18.9516413
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 0, lower bound: -18.9517610, upper bound: 18.9515778
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 0, lower bound: -18.9515778, upper bound: 18.9515778
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 0, lower bound: -18.9517459, upper bound: 18.9515634
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 0, lower bound: -18.9514723, upper bound: 18.9515478
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 0, lower bound: -18.9517531, upper bound: 18.9515647
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 0, lower bound: -18.9515447, upper bound: 18.9515514
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 0, lower bound: -18.9516867, upper bound: 18.9511704
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 0, lower bound: -18.9514743, upper bound: 18.9511704
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 0, lower bound: -18.9516867, upper bound: 18.9511704
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 0, lower bound: -18.9514743, upper bound: 18.9511704
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 0, lower bound: -18.9511528, upper bound: 18.9511528
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 0, lower bound: -18.9511528, upper bound: 18.9511528
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 0, lower bound: -18.9511528, upper bound: 18.9511528
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 0, lower bound: -18.9511528, upper bound: 18.9511528

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -9.3512793, 10.0293417, -9.4966640, 10.2702999, -19.6215782, 19.5260029
1: -75.8503876, 23.0836277, -78.2795029, 23.5250149, -99.3753967, 101.3631058
2: -39.0229263, 21.8521118, -39.9114876, 22.3295212, -61.3524437, 61.7635994
3: -51.5957794, 17.7047882, -53.1255913, 18.1230240, -69.7188034, 70.8303604
4: -27.0492325, 18.5571308, -27.5965233, 18.9551182, -46.0043373, 46.1536484

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513892, upper bound: 18.9512716
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515179, upper bound: 18.9513162
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -9.1126337, 9.8105164, -8.9571800, 9.6799469, -18.7925797, 18.7676926
1: -74.7118454, 22.5213013, -73.8682709, 22.1722736, -96.8841171, 96.3895721
2: -38.3192596, 21.3748131, -37.7859039, 21.0613575, -59.3806152, 59.1607170
3: -50.7557602, 17.3102989, -50.1258888, 17.0795021, -67.8352585, 67.4361877
4: -26.4188995, 18.1270275, -25.9968891, 17.8761425, -44.2950401, 44.1239128

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511196, upper bound: 18.9512734
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512777, upper bound: 18.9512973
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -11.1627655, 12.1892815, -9.6540794, 10.4347181, -21.5974827, 21.8433609
1: -92.3871231, 27.6228523, -79.3318558, 23.9139748, -116.3010864, 106.9547043
2: -47.1222954, 26.3352623, -40.5086136, 22.6908417, -69.8131409, 66.8438644
3: -62.6433334, 21.4003105, -53.8613892, 18.4144821, -81.0578156, 75.2617035
4: -32.4451828, 22.4504395, -28.0318661, 19.2741928, -51.7193756, 50.4823036

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513720, upper bound: 18.9512576
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513590, upper bound: 18.9512330
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -10.9891129, 12.0381832, -9.1871710, 9.9213552, -20.9104671, 21.2253532
1: -91.7071228, 27.2714214, -75.4456024, 22.7175980, -114.4247208, 102.7170258
2: -46.6865997, 26.0075703, -38.6633911, 21.5696259, -68.2562256, 64.6709595
3: -62.1220093, 21.1515484, -51.2161713, 17.4949360, -79.6169434, 72.3677063
4: -32.0371399, 22.1488953, -26.6209526, 18.3268528, -50.3639793, 48.7698402

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511196, upper bound: 18.9512609
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512330, upper bound: 18.9512330
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -9.3512793, 10.0293417, -16.7967587, 18.7255840, -28.0768623, 26.8260956
1: -75.8503876, 23.0836277, -132.5704041, 41.6031990, -117.4535828, 155.6539917
2: -39.0229263, 21.8521118, -68.5042953, 39.7008324, -78.7237549, 90.3564072
3: -51.5957794, 17.7047882, -90.3070374, 32.6720390, -84.2677841, 108.0118103
4: -27.0492325, 18.5571308, -47.9175301, 34.9279633, -61.9771843, 66.4746628

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511120, upper bound: 18.9510211
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9516126, upper bound: 18.9514289
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -9.1126337, 9.8105164, -16.1709995, 18.0515614, -27.1641960, 25.9815159
1: -74.7118454, 22.5213013, -128.0944061, 40.0725670, -114.7844086, 150.6157074
2: -38.3192596, 21.3748131, -66.1924896, 38.2787056, -76.5979614, 87.5673065
3: -50.7557602, 17.3102989, -87.2176285, 31.5002575, -82.2560196, 104.5279236
4: -26.4188995, 18.1270275, -46.1289902, 33.6784630, -60.0973625, 64.2560196

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510214, upper bound: 18.9510058
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513020, upper bound: 18.9513859
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -11.1627655, 12.1892815, -16.8915730, 18.8084335, -29.9711952, 29.0808525
1: -92.3871231, 27.6228523, -133.2178345, 41.8525658, -134.2396545, 160.8406830
2: -47.1222954, 26.3352623, -68.8660889, 39.9116325, -87.0339279, 95.2013474
3: -62.6433334, 21.4003105, -90.7664108, 32.8362198, -95.4795532, 112.1667175
4: -32.4451828, 22.4504395, -48.2002563, 35.0792503, -67.5244293, 70.6506958

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513614, upper bound: 18.9511250
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9516233, upper bound: 18.9514315
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -10.9891129, 12.0381832, -16.3372746, 18.2081661, -29.1972790, 28.3754539
1: -91.7071228, 27.2714214, -129.1864929, 40.4776764, -132.1847687, 156.4579163
2: -46.6865997, 26.0075703, -66.8256531, 38.6253166, -85.3119202, 92.8332138
3: -62.1220093, 21.1515484, -87.9967194, 31.7792931, -93.9013062, 109.1482620
4: -32.0371399, 22.1488953, -46.6035538, 33.9544067, -65.9915466, 68.7524490

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512422, upper bound: 18.9510596
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513020, upper bound: 18.9513700
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -16.7518044, 18.5883274, -10.2026196, 11.0001450, -27.7519474, 28.7909470
1: -130.8214111, 41.3927078, -81.1365585, 25.4654350, -156.2868347, 122.5292664
2: -67.9894409, 39.4433517, -42.2413712, 23.9027443, -91.8921738, 81.6847000
3: -89.2758255, 32.4117126, -55.5976448, 19.5095367, -108.7853622, 88.0093536
4: -47.6846008, 34.7120934, -29.5800400, 20.4724350, -68.1570358, 64.2921295

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515671, upper bound: 18.9515671
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515671, upper bound: 18.9515671
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -16.5600872, 18.4180260, -9.6426306, 10.4506483, -27.0107346, 28.0606575
1: -130.0539246, 40.9412766, -77.2980881, 23.9946995, -154.0485840, 118.2393570
2: -67.4849548, 39.0606689, -40.0057144, 22.6012058, -90.0861511, 79.0663757
3: -88.6934509, 32.1024933, -52.7640495, 18.4823303, -107.1757812, 84.8665466
4: -47.1657715, 34.3713989, -27.8986816, 19.4038105, -66.5695801, 62.2700806

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515671, upper bound: 18.9515671
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515671, upper bound: 18.9515671
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -48.8962631, 51.8594246, -10.2026196, 11.0001450, -59.2758026, 61.6180420
1: -343.1764526, 118.6750488, -81.1365585, 25.4654350, -359.1441040, 199.8115997
2: -190.3836212, 110.3479538, -42.2413712, 23.9027443, -209.3759918, 152.5893250
3: -239.6437683, 90.0085068, -55.5976448, 19.5095367, -252.0615997, 145.6061401
4: -137.4658813, 97.4595032, -29.5800400, 20.4724350, -153.8949738, 127.0246964

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514743, upper bound: 18.9511704
time: 0.86 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514743, upper bound: 18.9511704
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -48.5715790, 51.5438843, -9.6426306, 10.4506483, -58.4245110, 60.7491302
1: -341.8863220, 117.8697968, -77.2980881, 23.9946995, -356.3516235, 195.1678772
2: -189.2749329, 109.6429977, -40.0057144, 22.6012058, -207.0968781, 149.6487122
3: -238.5390625, 89.4329453, -52.7640495, 18.4823303, -250.0029297, 142.1969757
4: -136.6434326, 96.7727890, -27.8986816, 19.4038105, -152.1039581, 124.6714706

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514743, upper bound: 18.9511704
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514743, upper bound: 18.9511704
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -15.4601688, 17.1181679, -47.5265541, 49.8262787, -64.6656113, 63.6578903
1: -120.7814865, 38.1829414, -335.6467896, 115.5186157, -236.3000793, 363.1257629
2: -62.5406075, 36.3876457, -186.5644684, 106.7992096, -168.9407349, 216.6845245
3: -82.3589401, 29.8769817, -234.7189941, 86.7913666, -169.1502991, 256.2236328
4: -43.9425278, 31.9672260, -134.3626709, 93.1123428, -136.8911896, 161.2553558

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511544, upper bound: 18.9513851
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511544, upper bound: 18.9513851
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -15.4601688, 17.1181679, -47.2682190, 49.6497612, -64.4763870, 63.4532318
1: -120.7814865, 38.1829414, -334.2137451, 114.8112640, -235.5927429, 361.4869080
2: -62.5406075, 36.3876457, -185.4826813, 106.2768936, -168.3664856, 215.8124084
3: -82.3589401, 29.8769817, -233.5611725, 86.3922806, -168.7512054, 255.1020203
4: -43.9425278, 31.9672260, -133.6108398, 92.7838287, -136.4969025, 160.6782684

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511544, upper bound: 18.9513851
time: 0.84 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511544, upper bound: 18.9513851
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -14.9710941, 16.5930767, -47.4654961, 49.7578621, -64.0830231, 63.0541916
1: -117.1259918, 36.9314232, -335.2470093, 115.3721924, -232.1904144, 361.3952637
2: -60.6932373, 35.2148247, -186.3436737, 106.6533127, -166.7766571, 215.2074432
3: -79.8433533, 28.9404850, -234.4420166, 86.6740952, -166.5174408, 254.9641418
4: -42.5029488, 30.9753571, -134.1987610, 92.9759140, -135.2300110, 160.0643005

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511528, upper bound: 18.9511528
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511528, upper bound: 18.9511528
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -14.9710941, 16.5930767, -47.2153206, 49.5917397, -63.9315186, 62.8970718
1: -117.1259918, 36.9314232, -333.8688965, 114.6846466, -231.7436066, 360.0470581
2: -60.6932373, 35.2148247, -185.2903748, 106.1516876, -166.3420715, 214.5506897
3: -79.8433533, 28.9404850, -233.3220978, 86.2920761, -166.1354065, 254.0635529
4: -42.5029488, 30.9753571, -133.4680786, 92.6675262, -134.9333344, 159.6391449

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511528, upper bound: 18.9511528
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511528, upper bound: 18.9511528
time: 0.92 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.94 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -18.9513892, upper bound: 18.9512716
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -18.9515179, upper bound: 18.9513162
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -18.9511196, upper bound: 18.9512734
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -18.9512777, upper bound: 18.9512973
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -18.9513720, upper bound: 18.9512576
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -18.9513590, upper bound: 18.9512330
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -18.9511196, upper bound: 18.9512609
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -18.9512330, upper bound: 18.9512330
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -18.9511120, upper bound: 18.9510211
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -18.9516126, upper bound: 18.9514289
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -18.9510214, upper bound: 18.9510058
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -18.9513020, upper bound: 18.9513859
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -18.9513614, upper bound: 18.9511250
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -18.9516233, upper bound: 18.9514315
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -18.9512422, upper bound: 18.9510596
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -18.9513020, upper bound: 18.9513700
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -18.9515671, upper bound: 18.9515671
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -18.9515671, upper bound: 18.9515671
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -18.9515671, upper bound: 18.9515671
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -18.9515671, upper bound: 18.9515671
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -18.9514743, upper bound: 18.9511704
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -18.9514743, upper bound: 18.9511704
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -18.9514743, upper bound: 18.9511704
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -18.9514743, upper bound: 18.9511704
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -18.9511544, upper bound: 18.9513851
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -18.9511544, upper bound: 18.9513851
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -18.9511544, upper bound: 18.9513851
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -18.9511544, upper bound: 18.9513851
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -18.9511528, upper bound: 18.9511528
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -18.9511528, upper bound: 18.9511528
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -18.9511528, upper bound: 18.9511528
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 0, lower bound: -18.9511528, upper bound: 18.9511528

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -9.4302673, 10.1805925, -9.0567570, 9.8217525, -19.2520180, 19.2373447
1: -76.0816956, 23.3002186, -74.1955566, 22.4485188, -98.5302048, 97.4957733
2: -39.1999359, 22.0577812, -37.9068260, 21.2919617, -60.4918861, 59.9646034
3: -51.7756462, 17.9821758, -50.3739433, 17.3487453, -69.1243896, 68.3561172
4: -27.1819153, 18.8830967, -26.2440395, 18.1722927, -45.3542023, 45.1271362

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9508603, upper bound: 18.9510821
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512559, upper bound: 18.9511191
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -9.0344419, 9.6892653, -9.3171806, 10.0767736, -19.1112099, 19.0064468
1: -73.3835754, 22.2890205, -76.8941193, 23.0770702, -96.4606400, 99.1831284
2: -37.7187881, 21.1081810, -39.1801300, 21.9061451, -59.6249313, 60.2883034
3: -49.9013824, 17.1045799, -52.1746483, 17.7811222, -67.6825027, 69.2792282
4: -26.1342964, 17.9224834, -27.0809155, 18.5908852, -44.7251778, 45.0033989

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510379, upper bound: 18.9511323
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514081, upper bound: 18.9511810
time: 1.09 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -9.2017622, 9.9732656, -8.5208426, 9.2249489, -18.4267120, 18.4941063
1: -75.0855255, 22.8119545, -69.7787704, 21.0999355, -96.1854477, 92.5907211
2: -38.5697060, 21.6179104, -35.7695465, 20.0247040, -58.5944061, 57.3874588
3: -51.0220413, 17.6304626, -47.3619919, 16.3013783, -67.3234177, 64.9924545
4: -26.6037292, 18.4806366, -24.6478729, 17.0924778, -43.6961975, 43.1285095

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9508584, upper bound: 18.9510885
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9509585, upper bound: 18.9511183
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -8.8129454, 9.4888496, -8.7848005, 9.4939022, -18.3068428, 18.2736492
1: -72.3298874, 21.7768135, -72.5086365, 21.7449589, -94.0748291, 94.2854462
2: -37.0776367, 20.6693535, -37.0794182, 20.6562786, -57.7339096, 57.7487717
3: -49.1263351, 16.7447395, -49.1975822, 16.7522564, -65.8785858, 65.9423218
4: -25.5457020, 17.5295219, -25.5014400, 17.5302505, -43.0759506, 43.0309525

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510297, upper bound: 18.9511307
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511122, upper bound: 18.9511466
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -11.3697796, 12.4961319, -9.2194738, 9.9939394, -21.3637180, 21.7156067
1: -93.2392731, 28.1763382, -75.2164917, 22.8511314, -116.0903931, 103.3928299
2: -47.7272911, 26.8415203, -38.4848595, 21.6585960, -69.3858871, 65.3263779
3: -63.2558899, 21.9529457, -51.0944633, 17.6515884, -80.9074783, 73.0474014
4: -32.8938332, 23.1030922, -26.6913052, 18.5014896, -51.3953171, 49.7943916

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9508603, upper bound: 18.9510768
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512311, upper bound: 18.9511078
time: 1.52 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -10.9126167, 11.9196186, -9.4786816, 10.2420864, -21.1547031, 21.3983002
1: -90.4256668, 27.0031872, -77.9761963, 23.4776478, -113.9033127, 104.9793777
2: -46.0964813, 25.7466183, -39.7957306, 22.2743778, -68.3708572, 65.5423508
3: -61.3014755, 20.9298000, -52.9362259, 18.0756950, -79.3771515, 73.8660202
4: -31.7139053, 21.9481430, -27.5327740, 18.9117985, -50.6257019, 49.4809113

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9509591, upper bound: 18.9510702
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9509591, upper bound: 18.9510669
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -11.2060089, 12.3532772, -8.7555780, 9.4760447, -20.6820507, 21.1088562
1: -92.6203156, 27.8497677, -71.3285370, 21.6600533, -114.2803497, 99.1783066
2: -47.3349609, 26.5410461, -36.6490555, 20.5394249, -67.8743896, 63.1900978
3: -62.7813873, 21.7197514, -48.4417305, 16.7281990, -79.5095825, 70.1614761
4: -32.5349998, 22.8203526, -25.2817116, 17.5560322, -50.0910225, 48.1020660

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9509113, upper bound: 18.9510800
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9509585, upper bound: 18.9511046
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -10.7427988, 11.7726755, -9.0194187, 9.7366619, -20.4794617, 20.7920952
1: -89.7524719, 26.6571331, -74.1224136, 22.3018723, -112.0543442, 100.7795486
2: -45.6674767, 25.4295635, -37.9785347, 21.1717415, -66.8392181, 63.4080963
3: -60.7862396, 20.6870060, -50.3161659, 17.1721687, -77.9584045, 71.0031738
4: -31.3242149, 21.6579647, -26.1437912, 17.9829788, -49.3071899, 47.8017578

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9509571, upper bound: 18.9510702
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510086, upper bound: 18.9510558
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -8.4398508, 9.0485668, -16.4436512, 18.3820648, -26.8219147, 25.4922180
1: -68.3959122, 20.8184357, -130.3999939, 40.7492676, -109.1451569, 151.2183990
2: -35.2163353, 19.7442646, -67.2394333, 38.9344521, -74.1507874, 86.9836960
3: -46.5053444, 15.9973011, -88.7485962, 32.0594406, -78.5647888, 104.7458954
4: -24.3789120, 16.7987442, -46.9388504, 34.2741737, -58.6530762, 63.7375908

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9501727, upper bound: 18.9495358
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9502487, upper bound: 18.9495915
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -9.3162107, 9.9919834, -16.7600822, 18.6848850, -28.0010948, 26.7520618
1: -75.5990829, 23.0022354, -132.3129883, 41.5138054, -117.1128769, 155.3152161
2: -38.8920479, 21.7750912, -68.3610535, 39.6150246, -78.5070724, 90.1361313
3: -51.4230957, 17.6411686, -90.1265030, 32.6036453, -84.0267334, 107.7676697
4: -26.9546165, 18.4885654, -47.8151550, 34.8518181, -61.8064346, 66.3037186

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514824, upper bound: 18.9513425
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515734, upper bound: 18.9513882
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -8.1705580, 8.7908602, -15.8177052, 17.6959019, -25.8664589, 24.6085625
1: -66.9042206, 20.1645622, -125.8457565, 39.2288055, -106.1330261, 146.0103149
2: -34.3343277, 19.1786003, -64.9034882, 37.5057335, -71.8400574, 84.0820847
3: -45.4252777, 15.5374575, -85.6243362, 30.8799648, -76.3052216, 101.1617737
4: -23.6434727, 16.3043098, -45.1691437, 33.0072441, -56.6507187, 61.4734535

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9501439, upper bound: 18.9495090
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9501149, upper bound: 18.9495090
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -9.0699081, 9.7641573, -16.1478424, 18.0259552, -27.0958633, 25.9119968
1: -74.4148331, 22.4187469, -127.9354706, 40.0162773, -114.4311066, 150.3542023
2: -38.1588364, 21.2772522, -66.1063385, 38.2249260, -76.3837585, 87.3835831
3: -50.5498695, 17.2311096, -87.1071930, 31.4568615, -82.0067139, 104.3383026
4: -26.3011189, 18.0399151, -46.0655594, 33.6305084, -59.9316254, 64.1054688

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513020, upper bound: 18.9513363
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511805, upper bound: 18.9513542
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -10.2723675, 11.2348080, -16.5517559, 18.4807911, -28.7531586, 27.7865639
1: -85.1554413, 25.4291553, -131.1662903, 41.0265083, -126.1819458, 156.5954437
2: -43.4157104, 24.2816315, -67.6649475, 39.1752014, -82.5909119, 91.9465790
3: -57.6994095, 19.7466602, -89.2971420, 32.2469406, -89.9463501, 109.0437927
4: -29.8693714, 20.7393246, -47.2688026, 34.4515190, -64.3208923, 68.0081253

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512224, upper bound: 18.9510585
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9502487, upper bound: 18.9495915
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -11.1205072, 12.1426172, -16.8573475, 18.7704430, -29.8909492, 28.9999657
1: -92.0840454, 27.5207443, -132.9778748, 41.7691917, -133.8532410, 160.4985962
2: -46.9588165, 26.2390537, -68.7337265, 39.8310127, -86.7898254, 94.9727783
3: -62.4334755, 21.3211784, -90.5987625, 32.7721405, -95.2056122, 111.9199295
4: -32.3271942, 22.3645706, -48.1053581, 35.0076332, -67.3348236, 70.4699249

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9501727, upper bound: 18.9513402
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9516154, upper bound: 18.9513805
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -10.0706043, 11.0460978, -15.9792032, 17.8514271, -27.9220314, 27.0252991
1: -84.1499329, 24.9800682, -126.9208527, 39.6199112, -123.7698212, 151.9009094
2: -42.8147926, 23.8721294, -65.5207520, 37.8496132, -80.6644058, 89.3928604
3: -56.9576035, 19.4288177, -86.3866196, 31.1562920, -88.1138916, 105.8154373
4: -29.3397579, 20.3740921, -45.6310234, 33.2833061, -62.6230621, 66.0051117

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512275, upper bound: 18.9510422
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512422, upper bound: 18.9510596
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -10.9445887, 11.9891510, -16.3173275, 18.1860886, -29.1306705, 28.3064785
1: -91.3921890, 27.1637058, -129.0408936, 40.4286766, -131.8208618, 156.2045746
2: -46.5162354, 25.9058571, -66.7488556, 38.5792732, -85.0955048, 92.6547089
3: -61.9044228, 21.0683727, -87.8963089, 31.7415123, -93.6459274, 108.9646759
4: -31.9133759, 22.0581512, -46.5478210, 33.9138184, -65.8271866, 68.6059647

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513837, upper bound: 18.9513267
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513224, upper bound: 18.9513404
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -16.6295528, 18.5346375, -10.2026196, 11.0001450, -27.6296978, 28.7372570
1: -131.0569611, 41.1470451, -81.1365585, 25.4654350, -156.5223999, 122.2835999
2: -67.7690582, 39.2821808, -42.2413712, 23.9027443, -91.6717987, 81.5235138
3: -89.3016357, 32.3180122, -55.5976448, 19.5095367, -108.8111725, 87.9156570
4: -47.4180260, 34.5792618, -29.5800400, 20.4724350, -67.8904572, 64.1593018

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510628, upper bound: 18.9512106
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515853, upper bound: 18.9514367
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -16.0751400, 17.9412117, -10.2026196, 11.0001450, -27.0752831, 28.1438313
1: -126.9912262, 39.7951851, -81.1365585, 25.4654350, -152.4566650, 120.9317474
2: -65.7178040, 38.0067024, -42.2413712, 23.9027443, -89.6205139, 80.2480621
3: -86.5101852, 31.2754898, -55.5976448, 19.5095367, -106.0197220, 86.8731384
4: -45.8221207, 33.4684219, -29.5800400, 20.4724350, -66.2945557, 63.0484619

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510628, upper bound: 18.9512106
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515853, upper bound: 18.9514367
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -16.6295528, 18.5346375, -9.6426306, 10.4506483, -27.0802002, 28.1772690
1: -131.0569611, 41.1470451, -77.2980881, 23.9946995, -155.0516663, 118.4451294
2: -67.7690582, 39.2821808, -40.0057144, 22.6012058, -90.3702621, 79.2878799
3: -89.3016357, 32.3180122, -52.7640495, 18.4823303, -107.7839661, 85.0820618
4: -47.4180260, 34.5792618, -27.8986816, 19.4038105, -66.8218384, 62.4779396

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510620, upper bound: 18.9512106
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513691, upper bound: 18.9513691
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -16.0868149, 17.9531040, -9.6426306, 10.4506483, -26.5374584, 27.5957336
1: -127.0635071, 39.8241577, -77.2980881, 23.9946995, -151.0582123, 117.1222382
2: -65.7606888, 38.0339279, -40.0057144, 22.6012058, -88.3618927, 78.0396423
3: -86.5631638, 31.2962189, -52.7640495, 18.4823303, -105.0454941, 84.0602646
4: -45.8554192, 33.4918213, -27.8986816, 19.4038105, -65.2592316, 61.3904991

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510620, upper bound: 18.9512106
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513691, upper bound: 18.9513691
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -48.3714371, 51.3036995, -10.2026196, 11.0001450, -58.7269630, 61.0662422
1: -340.0133057, 117.3870163, -81.1365585, 25.4654350, -356.0229187, 198.5235748
2: -188.4087677, 109.1517487, -42.2413712, 23.9027443, -207.3739929, 151.3930969
3: -237.3364410, 89.0396805, -55.5976448, 19.5095367, -249.7283325, 144.6373291
4: -136.0473785, 96.3481140, -29.5800400, 20.4724350, -152.4435120, 125.9281387

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9508954, upper bound: 18.9507385
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514433, upper bound: 18.9509920
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -48.2356110, 51.2623787, -10.2026196, 11.0001450, -58.6280861, 61.0042572
1: -339.8934631, 117.0248795, -81.1365585, 25.4654350, -355.5160217, 198.1614380
2: -188.0272369, 108.9328537, -42.2413712, 23.9027443, -207.0437469, 151.1742249
3: -237.1137390, 88.8819122, -55.5976448, 19.5095367, -249.3568726, 144.4795380
4: -135.6925507, 96.2418747, -29.5800400, 20.4724350, -152.1945496, 125.7774963

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9508954, upper bound: 18.9507385
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514433, upper bound: 18.9509920
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -48.0932465, 50.9751129, -9.6426306, 10.4506483, -57.8772278, 60.1648254
1: -338.3117676, 116.7110748, -77.2980881, 23.9946995, -352.8193054, 194.0091553
2: -187.4467010, 108.4913483, -40.0057144, 22.6012058, -205.0322266, 148.4970703
3: -236.1363983, 88.4900131, -52.7640495, 18.4823303, -247.4500275, 141.2540436
4: -135.3170776, 95.6853409, -27.8986816, 19.4038105, -150.5957489, 123.5840225

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9508888, upper bound: 18.9507667
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511774, upper bound: 18.9509252
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -48.0471001, 51.0351715, -9.6426306, 10.4506483, -57.9053116, 60.2305641
1: -338.8207397, 116.5723648, -77.2980881, 23.9946995, -353.1355591, 193.8704529
2: -187.4055481, 108.4821930, -40.0057144, 22.6012058, -205.2048798, 148.4878998
3: -236.3537140, 88.5064774, -52.7640495, 18.4823303, -247.6687317, 141.2705231
4: -135.2089844, 95.7752762, -27.8986816, 19.4038105, -150.7067261, 123.6739578

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9508888, upper bound: 18.9507385
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511774, upper bound: 18.9509252
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -16.6143284, 18.5178509, -47.1496811, 49.4122200, -65.3699570, 64.6227188
1: -130.9620056, 41.1103363, -333.0158997, 114.6055756, -244.7314453, 363.2397156
2: -67.7101669, 39.2489624, -185.1389771, 105.8947449, -172.7893982, 217.9486389
3: -89.2332230, 32.2902412, -232.9066925, 86.0692749, -175.3024902, 256.6680908
4: -47.3759270, 34.5483894, -133.3249817, 92.2789459, -139.2760925, 162.7147217

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9506625, upper bound: 18.9508573
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514152, upper bound: 18.9508573
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -53.6824722, 57.0943604, -49.2961617, 51.8081856, -103.4547653, 104.2472076
1: -379.9264526, 130.5429688, -348.1985474, 119.8371582, -484.7106018, 464.5214844
2: -209.8783875, 121.6855392, -193.2834930, 111.1613693, -312.0566101, 305.9817810
3: -264.9808960, 99.1817245, -243.3537598, 90.2618637, -344.8234253, 332.2531738
4: -151.1158447, 107.4557877, -139.2274780, 97.2031631, -241.6570282, 239.5755310

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9506625, upper bound: 18.9508573
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514152, upper bound: 18.9514152
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -16.6143284, 18.5178509, -46.8990364, 49.2438507, -65.1879578, 64.4255219
1: -130.9620056, 41.1103363, -331.6398315, 113.9159622, -244.1501465, 361.6448364
2: -67.7101669, 39.2489624, -184.0875244, 105.3902893, -172.2296448, 217.1027985
3: -89.2332230, 32.2902412, -231.7869568, 85.6834259, -174.9166565, 255.5745697
4: -47.3759270, 34.5483894, -132.5954132, 91.9625854, -138.8953857, 162.1576233

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9506606, upper bound: 18.9508518
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9509664, upper bound: 18.9511467
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -53.6824722, 57.0943604, -48.9295502, 51.5034065, -103.1408539, 103.9326935
1: -379.9264526, 130.5429688, -345.8834839, 118.8759384, -483.8547974, 462.0512390
2: -209.8783875, 121.6855392, -191.7705536, 110.3737793, -311.2265625, 304.6770020
3: -264.9808960, 99.1817245, -241.6047211, 89.6515427, -344.1951294, 330.5604858
4: -151.1158447, 107.4557877, -138.1627808, 96.6145477, -241.0249176, 238.6847992

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9506606, upper bound: 18.9508518
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9509664, upper bound: 18.9511467
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -16.0751400, 17.9412117, -47.0792007, 49.3335991, -64.7360840, 63.9608307
1: -126.9912262, 39.7951851, -332.5493774, 114.4365387, -240.2647400, 361.3688965
2: -65.7178040, 38.0067024, -184.8820801, 105.7279434, -170.4182892, 216.3836823
3: -86.5101852, 31.2754898, -232.5840454, 85.9343948, -172.3244629, 255.2965240
4: -45.8221207, 33.4684219, -133.1353607, 92.1216965, -137.4808197, 161.4168701

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9505651, upper bound: 18.9507257
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511467, upper bound: 18.9509664
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -50.1325874, 53.2472191, -49.2961617, 51.8081856, -99.8908386, 100.3628693
1: -357.8932800, 121.8502731, -348.1985474, 119.8371582, -461.9934692, 455.8022766
2: -197.1344452, 113.4747086, -193.2834930, 111.1613693, -299.1865845, 297.7009277
3: -249.5372162, 92.5036240, -243.3537598, 90.2618637, -328.9287415, 325.5514526
4: -141.6385345, 99.8954315, -139.2274780, 97.2031631, -232.1815948, 232.0875702

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9505651, upper bound: 18.9507257
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511467, upper bound: 18.9509664
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -16.0751400, 17.9412117, -46.8351898, 49.1736641, -64.5935059, 63.8120117
1: -126.9912262, 39.7951851, -331.2102661, 113.7624359, -239.8700714, 360.0462036
2: -65.7178040, 38.0067024, -183.8513031, 105.2387924, -170.0046539, 215.7537537
3: -86.5101852, 31.2754898, -231.4901123, 85.5620346, -172.0388031, 254.4091492
4: -45.8221207, 33.4684219, -132.4220886, 91.8213654, -137.1982880, 161.0115967

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9505630, upper bound: 18.9507257
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9508431, upper bound: 18.9508431
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -50.1325874, 53.2472191, -48.9295502, 51.5034065, -99.6229553, 100.0942841
1: -357.8932800, 121.8502731, -345.8834839, 118.8759384, -461.4048157, 453.6029968
2: -197.1344452, 113.4747086, -191.7705536, 110.3737793, -298.5670471, 296.6079712
3: -249.5372162, 92.5036240, -241.6047211, 89.6515427, -328.4950562, 324.0549927
4: -141.6385345, 99.8954315, -138.1627808, 96.6145477, -231.6947632, 231.3436584

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9505630, upper bound: 18.9507257
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9508431, upper bound: 18.9508431
time: 0.76 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.50 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9508603, upper bound: 18.9510821
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9512559, upper bound: 18.9511191
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9510379, upper bound: 18.9511323
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9514081, upper bound: 18.9511810
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9508584, upper bound: 18.9510885
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9509585, upper bound: 18.9511183
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9510297, upper bound: 18.9511307
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9511122, upper bound: 18.9511466
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9508603, upper bound: 18.9510768
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9512311, upper bound: 18.9511078
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9509591, upper bound: 18.9510702
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9509591, upper bound: 18.9510669
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9509113, upper bound: 18.9510800
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9509585, upper bound: 18.9511046
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9509571, upper bound: 18.9510702
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9510086, upper bound: 18.9510558
NS_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9501727, upper bound: 18.9495358
NS_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9502487, upper bound: 18.9495915
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9514824, upper bound: 18.9513425
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9515734, upper bound: 18.9513882
NS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9501439, upper bound: 18.9495090
NS_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9501149, upper bound: 18.9495090
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9513020, upper bound: 18.9513363
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9511805, upper bound: 18.9513542
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9512224, upper bound: 18.9510585
NS_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9502487, upper bound: 18.9495915
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9501727, upper bound: 18.9513402
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9516154, upper bound: 18.9513805
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9512275, upper bound: 18.9510422
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9512422, upper bound: 18.9510596
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9513837, upper bound: 18.9513267
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9513224, upper bound: 18.9513404
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9510628, upper bound: 18.9512106
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9515853, upper bound: 18.9514367
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9510628, upper bound: 18.9512106
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9515853, upper bound: 18.9514367
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9510620, upper bound: 18.9512106
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9513691, upper bound: 18.9513691
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9510620, upper bound: 18.9512106
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9513691, upper bound: 18.9513691
NS_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9508954, upper bound: 18.9507385
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9514433, upper bound: 18.9509920
NS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9508954, upper bound: 18.9507385
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9514433, upper bound: 18.9509920
NS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9508888, upper bound: 18.9507667
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9511774, upper bound: 18.9509252
NS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9508888, upper bound: 18.9507385
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9511774, upper bound: 18.9509252
NS_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9506625, upper bound: 18.9508573
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9514152, upper bound: 18.9508573
NS_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9506625, upper bound: 18.9508573
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9514152, upper bound: 18.9514152
NS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9506606, upper bound: 18.9508518
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9509664, upper bound: 18.9511467
NS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9506606, upper bound: 18.9508518
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9509664, upper bound: 18.9511467
NS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9505651, upper bound: 18.9507257
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9511467, upper bound: 18.9509664
NS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9505651, upper bound: 18.9507257
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9511467, upper bound: 18.9509664
NS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9505630, upper bound: 18.9507257
NS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9508431, upper bound: 18.9508431
NS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9505630, upper bound: 18.9507257
NS_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.50
Output dim: 0, lower bound: -18.9508431, upper bound: 18.9508431

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -9.0460405, 9.8178720, -8.0972691, 8.7780447, -17.8240852, 17.9151421
1: -73.8227997, 22.4241409, -66.0902481, 20.0422649, -93.8650589, 88.5143890
2: -37.8843384, 21.2610359, -33.8127060, 19.0487003, -56.9330368, 55.0737419
3: -50.1607094, 17.3505764, -44.8601799, 15.5213976, -65.6821060, 62.2107544
4: -26.1597595, 18.1931629, -23.4196091, 16.3113785, -42.4711380, 41.6127701

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9508603, upper bound: 18.9510821
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9508603, upper bound: 18.9510821
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -9.4164438, 10.1655722, -8.9851027, 9.7422457, -19.1586876, 19.1506748
1: -75.9819107, 23.2671146, -73.6561127, 22.2734642, -98.2553711, 96.9232101
2: -39.1470146, 22.0266228, -37.6150093, 21.1230068, -60.2700195, 59.6416321
3: -51.7068520, 17.9566460, -49.9993706, 17.2131977, -68.9200516, 67.9560089
4: -27.1442623, 18.8552856, -26.0405006, 18.0236130, -45.1678772, 44.8957863

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512559, upper bound: 18.9511191
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512559, upper bound: 18.9511191
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.6606445, 9.3333197, -8.3676548, 9.0408468, -17.7014923, 17.7009735
1: -71.1806641, 21.3948555, -68.8742676, 20.6942558, -91.8749161, 90.2691193
2: -36.4247437, 20.3206749, -35.1447601, 19.6812649, -56.1060104, 55.4654312
3: -48.3267365, 16.4723301, -46.7226639, 15.9680195, -64.2947540, 63.1949921
4: -25.1205711, 17.2411594, -24.2863808, 16.7427769, -41.8633499, 41.5275421

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510347, upper bound: 18.9510277
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510379, upper bound: 18.9511323
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -9.0201044, 9.6742783, -9.2466106, 9.9979086, -19.0180092, 18.9208813
1: -73.2848969, 22.2563114, -76.3696899, 22.9043522, -96.1892395, 98.6259995
2: -37.6664352, 21.0772572, -38.8951607, 21.7398186, -59.4062538, 59.9724121
3: -49.8333740, 17.0790081, -51.8093147, 17.6469707, -67.4803391, 68.8883133
4: -26.0964565, 17.8948879, -26.8808670, 18.4436550, -44.5401039, 44.7757454

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511430, upper bound: 18.9509886
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513624, upper bound: 18.9511810
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -8.8334217, 9.6198444, -7.5647678, 8.1923676, -17.0257854, 17.1846123
1: -72.8444824, 21.9618416, -61.8059921, 18.7006969, -91.5451736, 83.7678299
2: -37.2635498, 20.8454971, -31.7046204, 17.7976303, -55.0611801, 52.5501175
3: -49.4216042, 17.0163002, -41.9235001, 14.5004988, -63.9221039, 58.9398003
4: -25.6354275, 17.8134422, -21.8227959, 15.2470884, -40.8825035, 39.6362381

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9508584, upper bound: 18.9510885
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9508584, upper bound: 18.9510821
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -9.1832619, 9.9536314, -8.4722681, 9.1727180, -18.3559799, 18.4258995
1: -74.9614639, 22.7684479, -69.4441910, 20.9841156, -95.9455566, 92.2126160
2: -38.5022049, 21.5768661, -35.5888863, 19.9144402, -58.4166374, 57.1657410
3: -50.9359894, 17.5969830, -47.1300621, 16.2118225, -67.1478119, 64.7270432
4: -26.5542450, 18.4439316, -24.5156803, 16.9938984, -43.5481339, 42.9596100

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9509585, upper bound: 18.9511183
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9509585, upper bound: 18.9511183
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -8.4550247, 9.1416569, -7.8389006, 8.4664679, -16.9214897, 16.9805565
1: -70.1213379, 20.9380894, -64.6110611, 19.3627300, -89.4840546, 85.5491486
2: -35.7878418, 19.9088821, -33.0582657, 18.4437599, -54.2316017, 52.9671478
3: -47.5482216, 16.1398373, -43.8113670, 14.9618521, -62.5100746, 59.9512024
4: -24.5792160, 16.8724442, -22.7041798, 15.6940517, -40.2732658, 39.5766220

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510164, upper bound: 18.9510416
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510297, upper bound: 18.9511307
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -8.7950411, 9.4695005, -8.7373514, 9.4427376, -18.2377777, 18.2068501
1: -72.2083282, 21.7342911, -72.1837540, 21.6319427, -93.8402634, 93.9180450
2: -37.0115395, 20.6287842, -36.9039688, 20.5485973, -57.5601349, 57.5327492
3: -49.0418320, 16.7119141, -48.9724388, 16.6647282, -65.7065582, 65.6843491
4: -25.4968910, 17.4932594, -25.3726463, 17.4336624, -42.9305496, 42.8659058

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510164, upper bound: 18.9509639
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511122, upper bound: 18.9511466
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -10.9985294, 12.1432629, -8.2709360, 8.9620838, -19.9606113, 20.4141998
1: -91.0794678, 27.3291149, -67.1549911, 20.4726467, -111.5521164, 94.4841080
2: -46.4513588, 26.0710487, -34.4323196, 19.4425774, -65.8939362, 60.5033684
3: -61.7008057, 21.3436813, -45.6168633, 15.8441610, -77.5449677, 66.9605179
4: -31.9250088, 22.4351959, -23.9085655, 16.6659527, -48.5909538, 46.3437576

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9509145, upper bound: 18.9510768
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9508603, upper bound: 18.9510768
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -11.3515139, 12.4760952, -9.1516590, 9.9180689, -21.2695827, 21.6277447
1: -93.1109619, 28.1324368, -74.6906586, 22.6855278, -115.7964935, 102.8230972
2: -47.6575127, 26.8002834, -38.2063103, 21.4979877, -69.1555023, 65.0065842
3: -63.1667709, 21.9191799, -50.7318001, 17.5226574, -80.6894073, 72.6509781
4: -32.8428345, 23.0663986, -26.4997253, 18.3606358, -51.2034683, 49.5661163

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512311, upper bound: 18.9511078
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512311, upper bound: 18.9511078
time: 1.53 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -10.5460272, 11.5699158, -8.5377665, 9.2160435, -19.7620697, 20.1076813
1: -88.2795639, 26.1557808, -69.9912872, 21.1165276, -109.3960648, 96.1470642
2: -44.8194580, 24.9778881, -35.7946892, 20.0698662, -64.8893280, 60.7725677
3: -59.7528305, 20.3227463, -47.5095062, 16.2787514, -76.0315857, 67.8322525
4: -30.7412014, 21.2828560, -24.7651558, 17.0837040, -47.8249016, 46.0480118

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9508603, upper bound: 18.9510702
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9509145, upper bound: 18.9510702
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -10.8955631, 11.9007530, -9.4104328, 10.1656132, -21.0611725, 21.3111801
1: -90.3039627, 26.9620743, -77.4555283, 23.3104992, -113.6144562, 104.4175873
2: -46.0309219, 25.7078781, -39.5186195, 22.1127834, -68.1437073, 65.2264938
3: -61.2173309, 20.8978825, -52.5757561, 17.9455833, -79.1629181, 73.4736404
4: -31.6665688, 21.9134045, -27.3382473, 18.7696209, -50.4361877, 49.2516518

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512175, upper bound: 18.9510669
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512175, upper bound: 18.9510669
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -10.8520947, 12.0021009, -7.8220558, 8.4642687, -19.3163624, 19.8241577
1: -90.4339447, 27.0003376, -63.4693909, 19.3185883, -109.7525101, 90.4697266
2: -46.0463104, 25.7722378, -32.6523743, 18.3615723, -64.4078827, 58.4246063
3: -61.2097778, 21.1059761, -43.0904961, 14.9659500, -76.1757278, 64.1964722
4: -31.5756626, 22.1556759, -22.5190487, 15.7537756, -47.3294373, 44.6747169

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9508584, upper bound: 18.9510801
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9508584, upper bound: 18.9510768
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -11.1860161, 12.3313322, -8.7044840, 9.4208345, -20.6068497, 21.0358162
1: -92.4812317, 27.8013458, -70.9688568, 21.5376644, -114.0188904, 98.7701874
2: -47.2592545, 26.4954796, -36.4561386, 20.4227924, -67.6820374, 62.9516182
3: -62.6847458, 21.6825790, -48.1934624, 16.6335583, -79.3183060, 69.8760376
4: -32.4797325, 22.7796860, -25.1412945, 17.4522991, -49.9320297, 47.9209824

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9508584, upper bound: 18.9511046
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9509585, upper bound: 18.9511046
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -10.3846254, 11.4248075, -8.0926666, 8.7280073, -19.1126289, 19.5174751
1: -87.5834961, 25.8116188, -66.3292770, 19.9757500, -107.5592346, 92.1408920
2: -44.3828506, 24.6662178, -34.0170326, 19.0048447, -63.3876953, 58.6832466
3: -59.2262306, 20.0774269, -45.0105133, 15.4211102, -74.6473160, 65.0879288
4: -30.3649693, 20.9978104, -23.4079552, 16.1881275, -46.5530891, 44.4057579

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9509571, upper bound: 18.9510702
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9509571, upper bound: 18.9510702
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -10.7237501, 11.7518444, -8.9693785, 9.6825542, -20.4063034, 20.7212219
1: -89.6204834, 26.6111488, -73.7739334, 22.1821537, -111.8026199, 100.3850784
2: -45.5954590, 25.3862762, -37.7910423, 21.0575752, -66.6530304, 63.1773148
3: -60.6946144, 20.6516285, -50.0748749, 17.0794697, -77.7740784, 70.7265015
4: -31.2716389, 21.6192513, -26.0067558, 17.8810406, -49.1526756, 47.6260071

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510558, upper bound: 18.9510558
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510558, upper bound: 18.9510558
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -9.2267809, 9.8943739, -17.1643276, 18.9959469, -28.2227268, 27.0587006
1: -74.9233398, 22.7813034, -134.0053101, 42.5464020, -117.4697342, 156.7866211
2: -38.5287056, 21.5637741, -69.6105423, 40.4606934, -78.9893951, 91.1743088
3: -50.9564629, 17.4700813, -91.4367371, 33.2539940, -84.2104340, 108.9068146
4: -26.6991959, 18.3034878, -48.8909836, 35.5345116, -62.2336960, 67.1944733

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514824, upper bound: 18.9513425
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514824, upper bound: 18.9513425
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -9.2957973, 9.9677124, -16.5146523, 18.3910122, -27.6868076, 26.4823647
1: -75.3874588, 22.9497528, -129.9214783, 40.8834152, -116.2708588, 152.8712311
2: -38.7954903, 21.7230530, -67.1983643, 39.0036812, -77.7991714, 88.9214096
3: -51.2821159, 17.5985069, -88.5325241, 32.0895576, -83.3716736, 106.1310272
4: -26.8904018, 18.4456730, -47.0642281, 34.3295746, -61.2199707, 65.5099030

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515734, upper bound: 18.9513882
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515734, upper bound: 18.9513882
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -8.9834709, 9.6681299, -16.6308346, 18.4497051, -27.4331760, 26.2989655
1: -73.7459717, 22.2047062, -130.3395081, 41.1655655, -114.9115067, 152.5442047
2: -37.8077888, 21.0721626, -67.5954514, 39.2088547, -77.0166473, 88.6675949
3: -50.0924988, 17.0624886, -88.8668289, 32.2375298, -82.3300171, 105.9293060
4: -26.0543556, 17.8592930, -47.3230476, 34.4658813, -60.5202370, 65.1823425

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513020, upper bound: 18.9513363
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513020, upper bound: 18.9513363
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -9.0481787, 9.7384806, -15.8789091, 17.7141514, -26.7623291, 25.6173878
1: -74.2027435, 22.3606148, -125.3958664, 39.3143616, -113.5170975, 147.7564850
2: -38.0558853, 21.2222919, -64.8722153, 37.5538025, -75.6096802, 86.0944977
3: -50.4073448, 17.1853218, -85.4074707, 30.8999691, -81.3073120, 102.5927734
4: -26.2343483, 17.9943085, -45.2377319, 33.0657272, -59.3000717, 63.2320404

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511805, upper bound: 18.9513542
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511805, upper bound: 18.9513542
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -10.1620855, 11.1125298, -16.9193344, 18.7647038, -28.9267883, 28.0318642
1: -84.2876740, 25.1562386, -132.6755219, 41.9401970, -126.2278671, 157.8317566
2: -42.9592476, 24.0235729, -68.7478790, 39.9300842, -82.8893127, 92.7714539
3: -57.1049042, 19.5328407, -90.4536667, 32.8234520, -89.9283600, 109.9865036
4: -29.5498695, 20.5119343, -48.2453651, 35.0621300, -64.6119995, 68.7572861

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511292, upper bound: 18.9510172
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511292, upper bound: 18.9510172
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -11.0324211, 12.0448284, -17.2466202, 19.0645542, -30.0969753, 29.2914486
1: -91.4118652, 27.3021145, -134.5663452, 42.7659569, -134.1778259, 161.8684387
2: -46.6023140, 26.0306339, -69.9713135, 40.6418190, -87.2441177, 96.0019379
3: -61.9724998, 21.1491547, -91.8470840, 33.3984871, -95.3709717, 112.9962387
4: -32.0766335, 22.1799412, -49.1404953, 35.6593971, -67.7360306, 71.3204346

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515139, upper bound: 18.9513402
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515139, upper bound: 18.9513402
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -11.1006899, 12.1179485, -16.6148872, 18.4805241, -29.5812092, 28.7328300
1: -91.8749084, 27.4666061, -130.6161041, 41.1457024, -133.0206146, 158.0827026
2: -46.8591843, 26.1873798, -67.5861740, 39.2278061, -86.0869675, 93.7735519
3: -62.2944069, 21.2777386, -89.0193024, 32.2641449, -94.5585480, 110.2970352
4: -32.2668648, 22.3216858, -47.3677483, 34.4921570, -66.7590179, 69.6894379

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9516154, upper bound: 18.9513805
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9516154, upper bound: 18.9513805
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -9.9600143, 10.9234362, -16.4357796, 18.2549267, -28.2149353, 27.3592148
1: -83.2775574, 24.7079468, -129.2282562, 40.7048225, -123.9823761, 153.9361877
2: -42.3572273, 23.6120243, -66.9584961, 38.7732735, -81.1304855, 90.5705109
3: -56.3604736, 19.2146187, -88.0740967, 31.8990040, -88.2594757, 107.2887115
4: -29.0212536, 20.1454391, -46.8244400, 34.0681763, -63.0894318, 66.9698792

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511292, upper bound: 18.9510172
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511292, upper bound: 18.9510422
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -10.0536747, 11.0254726, -15.7188005, 17.5486736, -27.6023483, 26.7442741
1: -83.9836655, 24.9366779, -124.4334335, 38.9455109, -122.9291687, 149.3701019
2: -42.7351189, 23.8301487, -64.3294754, 37.2066650, -79.9417877, 88.1596222
3: -56.8477173, 19.3931751, -84.7333221, 30.6196785, -87.4673691, 104.1264954
4: -29.2878399, 20.3380299, -44.8326492, 32.7405777, -62.0284042, 65.1706772

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512333, upper bound: 18.9510596
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512333, upper bound: 18.9510596
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -10.8584747, 11.8926697, -16.7990208, 18.6064835, -29.4649563, 28.6916904
1: -90.7330322, 26.9502468, -131.3985596, 41.5926018, -132.3256378, 158.3488007
2: -46.1684608, 25.7007408, -68.2517929, 39.5672798, -85.7357407, 93.9525299
3: -61.4523506, 20.8986797, -89.6142426, 32.5290947, -93.9814301, 110.5129242
4: -31.6683426, 21.8751240, -47.8142319, 34.7507019, -66.4190445, 69.6893311

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513837, upper bound: 18.9513267
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513837, upper bound: 18.9513267
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -10.9240742, 11.9646702, -16.0580521, 17.8832245, -28.8072987, 28.0227222
1: -91.1905899, 27.1094875, -126.5486908, 39.7500267, -130.9406128, 153.6581726
2: -46.4190903, 25.8538284, -65.5410385, 37.9293823, -84.3484726, 91.3948669
3: -61.7697029, 21.0250397, -86.2265930, 31.2012348, -92.9709244, 107.2516327
4: -31.8486671, 22.0147190, -45.7435799, 33.3678894, -65.2165527, 67.7582932

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513224, upper bound: 18.9513404
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513224, upper bound: 18.9513404
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -16.2815628, 18.1974068, -9.2231197, 9.9294329, -26.2109909, 27.4205265
1: -128.9494934, 40.3110962, -72.8594818, 23.0201950, -151.9696960, 113.1705780
2: -66.5391693, 38.5293541, -38.1081123, 21.6212139, -88.1603851, 76.6374512
3: -87.7881088, 31.7167969, -49.9949226, 17.6404324, -105.4285431, 81.7117157
4: -46.4617195, 33.9337158, -26.7355232, 18.5649757, -65.0266953, 60.6692390

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9509532, upper bound: 18.9511883
time: 0.96 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510240, upper bound: 18.9513282
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -16.5989609, 18.5001373, -10.1306801, 10.9164104, -27.5153656, 28.6308136
1: -130.8344421, 41.0718956, -80.5604858, 25.2903690, -156.1248016, 121.6323700
2: -67.6471405, 39.2098045, -41.9556541, 23.7316723, -91.3788071, 81.1654434
3: -89.1466904, 32.2598000, -55.2167435, 19.3664436, -108.5131226, 87.4765396
4: -47.3318291, 34.5149803, -29.3776340, 20.3175545, -67.6493759, 63.8926125

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517505, upper bound: 18.9516947
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517807, upper bound: 18.9517807
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -15.7178669, 17.5844078, -9.2231197, 9.9294329, -25.6472931, 26.8075275
1: -124.7257462, 38.9436455, -72.8594818, 23.0201950, -147.7459412, 111.8031235
2: -64.4116440, 37.2307587, -38.1081123, 21.6212139, -86.0328522, 75.3388443
3: -84.9005051, 30.6533871, -49.9949226, 17.6404324, -102.5409317, 80.6483078
4: -44.8505363, 32.7962341, -26.7355232, 18.5649757, -63.4155121, 59.5317574

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9509994, upper bound: 18.9511935
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510240, upper bound: 18.9512106
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -16.0552750, 17.9191971, -10.1306801, 10.9164104, -26.9716854, 28.0498734
1: -126.8477173, 39.7467232, -80.5604858, 25.2903690, -152.1380920, 120.3072052
2: -65.6416092, 37.9609909, -41.9556541, 23.7316723, -89.3732681, 79.9166336
3: -86.4109879, 31.2379951, -55.2167435, 19.3664436, -105.7774277, 86.4547272
4: -45.7668571, 33.4278488, -29.3776340, 20.3175545, -66.0844040, 62.8054810

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514808, upper bound: 18.9510851
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9515853, upper bound: 18.9514219
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -16.2815628, 18.1974068, -8.6772985, 9.3969994, -25.6785622, 26.8747063
1: -128.9494934, 40.3110962, -69.1025925, 21.6134739, -150.5629730, 109.4136810
2: -66.5391693, 38.5293541, -35.9221573, 20.3474007, -86.8865662, 74.4515076
3: -87.7881088, 31.7167969, -47.2283325, 16.6627979, -104.4509048, 78.9451294
4: -46.4617195, 33.9337158, -25.0660610, 17.5322018, -63.9939194, 58.9997787

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510181, upper bound: 18.9511929
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510622, upper bound: 18.9513282
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -16.5989609, 18.5001373, -9.5923901, 10.3946323, -26.9935932, 28.0925274
1: -130.8344421, 41.0718956, -76.9119797, 23.8737431, -154.7081757, 117.9838562
2: -67.6471405, 39.2098045, -39.8135719, 22.4853668, -90.1324997, 79.0233612
3: -89.1466904, 32.2598000, -52.5075989, 18.3869743, -107.5336609, 84.7673950
4: -47.3318291, 34.5149803, -27.7592430, 19.3017921, -66.6336136, 62.2742233

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513105, upper bound: 18.9514400
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513374, upper bound: 18.9515433
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -15.7293901, 17.5962696, -8.6772985, 9.3969994, -25.1263885, 26.2735672
1: -124.7970734, 38.9725037, -69.1025925, 21.6134739, -146.4105530, 108.0750961
2: -64.4540329, 37.2577095, -35.9221573, 20.3474007, -84.8014374, 73.1798706
3: -84.9530792, 30.6740856, -47.2283325, 16.6627979, -101.6158676, 77.9024124
4: -44.8833427, 32.8195648, -25.0660610, 17.5322018, -62.4155426, 57.8856277

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510363, upper bound: 18.9511991
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510363, upper bound: 18.9512106
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -16.0669537, 17.9310894, -9.5923901, 10.3946323, -26.4615860, 27.5234795
1: -126.9199524, 39.7756996, -76.9119797, 23.8737431, -150.7937012, 116.6876831
2: -65.6845245, 37.9882202, -39.8135719, 22.4853668, -88.1698837, 77.8017883
3: -86.4639435, 31.2587242, -52.5075989, 18.3869743, -104.8509216, 83.7663269
4: -45.8001862, 33.4512444, -27.7592430, 19.3017921, -65.1019745, 61.2104874

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510175, upper bound: 18.9513621
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512859, upper bound: 18.9512859
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -48.1994972, 51.1152878, -10.1306801, 10.9164104, -58.4500160, 60.7986336
1: -338.7909546, 116.9589615, -80.5604858, 25.2903690, -354.5444336, 197.5194397
2: -187.7360535, 108.7489166, -41.9556541, 23.7316723, -206.4768524, 150.7045746
3: -236.4848328, 88.7099609, -55.2167435, 19.3664436, -248.6823730, 143.9266815
4: -135.5713196, 95.9842834, -29.3776340, 20.3175545, -151.7892609, 125.3398132

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9516378, upper bound: 18.9514319
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9517544, upper bound: 18.9514514
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -48.0765953, 51.0881310, -10.1306801, 10.9164104, -58.3640862, 60.7503815
1: -338.7680664, 116.6306686, -80.5604858, 25.2903690, -354.1312256, 197.1911011
2: -187.4129486, 108.5625916, -41.9556541, 23.7316723, -206.2000122, 150.5182495
3: -236.3310394, 88.5772858, -55.2167435, 19.3664436, -248.3737946, 143.7940063
4: -135.2534485, 95.9057007, -29.3776340, 20.3175545, -151.5767670, 125.1909409

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512535, upper bound: 18.9507017
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512535, upper bound: 18.9509920
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -47.9233017, 50.7874184, -9.5923901, 10.3946323, -57.6279221, 59.9187469
1: -337.0951233, 116.2881317, -76.9119797, 23.8737431, -351.3965759, 193.2001038
2: -186.7835693, 108.0925140, -39.8135719, 22.4853668, -204.1956787, 147.9060822
3: -235.2911987, 88.1636047, -52.5075989, 18.3869743, -246.4527740, 140.6712036
4: -134.8454437, 95.3245163, -27.7592430, 19.3017921, -149.9941559, 123.0515976

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9509919, upper bound: 18.9510257
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513113, upper bound: 18.9512589
time: 1.10 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -47.8895607, 50.8619232, -9.5923901, 10.3946323, -57.6683655, 59.9984245
1: -337.6981812, 116.1818466, -76.9119797, 23.8737431, -351.8027344, 193.0938263
2: -186.7981873, 108.1139450, -39.8135719, 22.4853668, -204.4192810, 147.9275208
3: -235.5756073, 88.2044144, -52.5075989, 18.3869743, -246.7326508, 140.7120056
4: -134.7730103, 95.4413223, -27.7592430, 19.3017921, -150.1405029, 123.1736984

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510737, upper bound: 18.9506450
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510737, upper bound: 18.9509252
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -16.5852909, 18.4851055, -47.0492020, 49.3010292, -65.2138596, 64.4758224
1: -130.7489166, 41.0391006, -332.3494263, 114.3618774, -244.1248932, 362.3223877
2: -67.5943298, 39.1799622, -184.7619629, 105.6603622, -172.3681030, 217.4194336
3: -89.0850754, 32.2349892, -232.4384766, 85.8804016, -174.9654694, 256.0214539
4: -47.2940140, 34.4873276, -133.0509186, 92.0654526, -138.9411011, 162.3426208

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514049, upper bound: 18.9516255
time: 0.92 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514218, upper bound: 18.9517136
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -53.6565361, 57.0653229, -49.2320137, 51.7373810, -103.3441772, 104.1455383
1: -379.7470093, 130.4798584, -347.7616882, 119.6812592, -484.2821045, 463.8663330
2: -209.7784729, 121.6248245, -193.0368042, 111.0113754, -311.7574463, 305.6076050
3: -264.8558960, 99.1324997, -243.0480042, 90.1414719, -344.5143127, 331.7917175
4: -151.0438995, 107.4014664, -139.0500641, 97.0700760, -241.4263916, 239.3162689

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512928, upper bound: 18.9513972
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9514077, upper bound: 18.9514077
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -16.5852909, 18.4851055, -46.8296051, 49.1662445, -65.0656052, 64.3107376
1: -130.7489166, 41.0391006, -331.1510925, 113.7471848, -243.6215973, 360.9151306
2: -67.5943298, 39.1799622, -183.8226624, 105.2281113, -171.8796387, 216.6916656
3: -89.0850754, 32.2349892, -231.4488373, 85.5513077, -174.6363831, 255.0634460
4: -47.2940140, 34.4873276, -132.4050751, 91.8154831, -138.6256409, 161.8745117

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9507235, upper bound: 18.9511622
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9506585, upper bound: 18.9513054
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -53.6565361, 57.0653229, -48.8943405, 51.4642067, -103.0621796, 103.8607864
1: -379.7470093, 130.4798584, -345.6256104, 118.7906799, -483.4990234, 461.5821533
2: -209.7784729, 121.6248245, -191.6335754, 110.2913895, -310.9944763, 304.4169006
3: -264.8558960, 99.1324997, -241.4277802, 89.5845642, -343.9397888, 330.2335205
4: -151.0438995, 107.4014664, -138.0652313, 96.5420837, -240.8545990, 238.5092163

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9507257, upper bound: 18.9505651
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9507257, upper bound: 18.9511467
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -16.0552750, 17.9191971, -46.9765053, 49.2199249, -64.5865402, 63.8222542
1: -126.8477173, 39.7467232, -331.8706970, 114.1876297, -239.7235718, 360.4602661
2: -65.6416092, 37.9609909, -184.4980011, 105.4885101, -170.0298920, 215.8691864
3: -86.4109879, 31.2379951, -232.1073914, 85.7413864, -171.9352875, 254.6585846
4: -45.7668571, 33.4278488, -132.8558044, 91.9031677, -137.1666565, 161.0593414

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512589, upper bound: 18.9512900
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511273, upper bound: 18.9511702
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -50.1165619, 53.2294044, -49.2320137, 51.7373810, -99.7904892, 100.2707596
1: -357.7733459, 121.8110199, -347.7616882, 119.6812592, -461.6037903, 455.1679382
2: -197.0719452, 113.4372864, -193.0368042, 111.0113754, -298.9161987, 297.3459778
3: -249.4555054, 92.4732132, -243.0480042, 90.1414719, -328.6501465, 325.1058350
4: -141.5938416, 99.8629913, -139.0500641, 97.0700760, -231.9736633, 231.8470001

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -18.9508518, upper bound: 18.9506606
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9508518, upper bound: 18.9509664
time: 0.73 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.32 seconds
NS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9508603, upper bound: 18.9510821
NS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9508603, upper bound: 18.9510821
NS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9512559, upper bound: 18.9511191
NS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9512559, upper bound: 18.9511191
NS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9510347, upper bound: 18.9510277
NS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9510379, upper bound: 18.9511323
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9511430, upper bound: 18.9509886
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9513624, upper bound: 18.9511810
NS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9508584, upper bound: 18.9510885
NS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9508584, upper bound: 18.9510821
NS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9509585, upper bound: 18.9511183
NS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9509585, upper bound: 18.9511183
NS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9510164, upper bound: 18.9510416
NS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9510297, upper bound: 18.9511307
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9510164, upper bound: 18.9509639
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9511122, upper bound: 18.9511466
NS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9509145, upper bound: 18.9510768
NS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9508603, upper bound: 18.9510768
NS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9512311, upper bound: 18.9511078
NS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9512311, upper bound: 18.9511078
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9508603, upper bound: 18.9510702
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9509145, upper bound: 18.9510702
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9512175, upper bound: 18.9510669
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9512175, upper bound: 18.9510669
NS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9508584, upper bound: 18.9510801
NS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9508584, upper bound: 18.9510768
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9508584, upper bound: 18.9511046
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9509585, upper bound: 18.9511046
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9509571, upper bound: 18.9510702
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9509571, upper bound: 18.9510702
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9510558, upper bound: 18.9510558
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9510558, upper bound: 18.9510558
NS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9514824, upper bound: 18.9513425
NS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9514824, upper bound: 18.9513425
NS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9515734, upper bound: 18.9513882
NS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9515734, upper bound: 18.9513882
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9513020, upper bound: 18.9513363
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9513020, upper bound: 18.9513363
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9511805, upper bound: 18.9513542
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9511805, upper bound: 18.9513542
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9511292, upper bound: 18.9510172
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9511292, upper bound: 18.9510172
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9515139, upper bound: 18.9513402
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9515139, upper bound: 18.9513402
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9516154, upper bound: 18.9513805
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9516154, upper bound: 18.9513805
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9511292, upper bound: 18.9510172
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9511292, upper bound: 18.9510422
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9512333, upper bound: 18.9510596
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9512333, upper bound: 18.9510596
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9513837, upper bound: 18.9513267
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9513837, upper bound: 18.9513267
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9513224, upper bound: 18.9513404
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9513224, upper bound: 18.9513404
NS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9509532, upper bound: 18.9511883
NS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9510240, upper bound: 18.9513282
NS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9517505, upper bound: 18.9516947
NS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9517807, upper bound: 18.9517807
NS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9509994, upper bound: 18.9511935
NS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9510240, upper bound: 18.9512106
NS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9514808, upper bound: 18.9510851
NS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9515853, upper bound: 18.9514219
NS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9510181, upper bound: 18.9511929
NS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9510622, upper bound: 18.9513282
NS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9513105, upper bound: 18.9514400
NS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9513374, upper bound: 18.9515433
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9510363, upper bound: 18.9511991
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9510363, upper bound: 18.9512106
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9510175, upper bound: 18.9513621
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9512859, upper bound: 18.9512859
NS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9516378, upper bound: 18.9514319
NS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9517544, upper bound: 18.9514514
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9512535, upper bound: 18.9507017
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9512535, upper bound: 18.9509920
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9509919, upper bound: 18.9510257
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9513113, upper bound: 18.9512589
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9510737, upper bound: 18.9506450
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9510737, upper bound: 18.9509252
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9514049, upper bound: 18.9516255
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9514218, upper bound: 18.9517136
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9512928, upper bound: 18.9513972
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9514077, upper bound: 18.9514077
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9507235, upper bound: 18.9511622
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9506585, upper bound: 18.9513054
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9507257, upper bound: 18.9505651
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9507257, upper bound: 18.9511467
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9512589, upper bound: 18.9512900
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9511273, upper bound: 18.9511702
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9508518, upper bound: 18.9506606
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 0, lower bound: -18.9508518, upper bound: 18.9509664

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -8.9600773, 9.8221169, -8.0972691, 8.7780447, -17.7381210, 17.9193859
1: -74.6404800, 22.3084240, -66.0902481, 20.0422649, -94.6827316, 88.3986664
2: -37.8886948, 21.2253304, -33.8127060, 19.0487003, -56.9373932, 55.0380363
3: -50.5386887, 17.3654995, -44.8601799, 15.5213976, -66.0600891, 62.2256775
4: -26.0495586, 18.1606846, -23.4196091, 16.3113785, -42.3609390, 41.5802917

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9508504, upper bound: 18.9510806
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9508504, upper bound: 18.9510821
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -8.3376856, 9.1244307, -8.0972691, 8.7780447, -17.1157303, 17.2216988
1: -69.7127380, 20.7716827, -66.0902481, 20.0422649, -89.7549973, 86.8619232
2: -35.4181366, 19.7648678, -33.8127060, 19.0487003, -54.4668350, 53.5775757
3: -47.1776009, 16.1546783, -44.8601799, 15.5213976, -62.6989975, 61.0148544
4: -24.2655697, 16.8753262, -23.4196091, 16.3113785, -40.5769501, 40.2949371

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9508504, upper bound: 18.9510806
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9508504, upper bound: 18.9510821
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -9.2812138, 10.1229925, -8.9851027, 9.7422457, -19.0234566, 19.1080952
1: -76.4536362, 23.0601673, -73.6561127, 22.2734642, -98.7270966, 96.7162552
2: -38.9970322, 21.8934002, -37.6150093, 21.1230068, -60.1200409, 59.5084076
3: -51.8654976, 17.8921623, -49.9993706, 17.2131977, -69.0786896, 67.8915329
4: -26.9062939, 18.7320652, -26.0405006, 18.0236130, -44.9299088, 44.7725639

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512559, upper bound: 18.9510690
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512234, upper bound: 18.9511191
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -8.6733189, 9.4518604, -8.9851027, 9.7422457, -18.4155598, 18.4369621
1: -71.7629929, 21.5673923, -73.6561127, 22.2734642, -94.0364532, 95.2235031
2: -36.6312981, 20.4828300, -37.6150093, 21.1230068, -57.7543030, 58.0978394
3: -48.6514740, 16.7287254, -49.9993706, 17.2131977, -65.8646622, 66.7280960
4: -25.1728363, 17.4970837, -26.0405006, 18.0236130, -43.1964493, 43.5375824

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512559, upper bound: 18.9510690
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9508603, upper bound: 18.9511191
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7.4683566, 7.9918857, -8.0698347, 8.6558952, -16.1242504, 16.0617199
1: -60.4985847, 18.3893776, -65.5188904, 19.9379635, -80.4365387, 83.9082489
2: -31.2393837, 17.4324017, -33.7306366, 18.9044571, -50.1438332, 51.1630363
3: -41.1570702, 14.1246328, -44.5825996, 15.3170099, -56.4740677, 58.7072220
4: -21.5639172, 14.8193388, -23.3600025, 16.0812340, -37.6451492, 38.1793404

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510347, upper bound: 18.9510277
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510347, upper bound: 18.9510277
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8.5419102, 9.2049894, -8.3341122, 9.0086031, -17.5505123, 17.5390987
1: -70.4723663, 21.1118793, -68.6883087, 20.6043301, -91.0766830, 89.8001862
2: -36.0146179, 20.0534058, -35.0307121, 19.6089802, -55.6235962, 55.0841179
3: -47.8296280, 16.2494984, -46.5892258, 15.9077377, -63.7373619, 62.8387222
4: -24.8068371, 16.9893436, -24.1911888, 16.6796265, -41.4864616, 41.1805305

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510379, upper bound: 18.9511323
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9508504, upper bound: 18.9511323
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7.7547207, 8.2765799, -8.9426527, 9.6045618, -17.3592777, 17.2192307
1: -62.3357430, 19.0755634, -72.9819717, 22.1164074, -84.4521484, 92.0575256
2: -32.2817383, 18.0554676, -37.4161415, 20.9353714, -53.2171097, 55.4716110
3: -42.4501648, 14.6230040, -49.6077194, 16.9709854, -59.4211502, 64.2307205
4: -22.3407478, 15.3614645, -25.9194946, 17.7603188, -40.1010513, 41.2809601

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511430, upper bound: 18.9509886
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511430, upper bound: 18.9509886
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.8998404, 9.5449352, -9.2218199, 9.9728212, -18.8726616, 18.7667542
1: -72.5681000, 21.9714527, -76.2134781, 22.8392448, -95.4073334, 98.1849289
2: -37.2484016, 20.8059578, -38.8046379, 21.6849613, -58.9333649, 59.6105957
3: -49.3276367, 16.8580399, -51.6972198, 17.6019211, -66.9295425, 68.5552597
4: -25.7792931, 17.6406059, -26.8129005, 18.3954945, -44.1747894, 44.4535065

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513624, upper bound: 18.9511810
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9513624, upper bound: 18.9511810
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -8.9600773, 9.8221169, -7.5647678, 8.1923676, -17.1524410, 17.3868847
1: -74.6404800, 22.3084240, -61.8059921, 18.7006969, -93.3411636, 84.1143951
2: -37.8886948, 21.2253304, -31.7046204, 17.7976303, -55.6863251, 52.9299469
3: -50.5386887, 17.3654995, -41.9235001, 14.5004988, -65.0391846, 59.2889938
4: -26.0495586, 18.1606846, -21.8227959, 15.2470884, -41.2966461, 39.9834785

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9508501, upper bound: 18.9510885
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9508501, upper bound: 18.9510885
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -8.3376856, 9.1244307, -7.5647678, 8.1923676, -16.5300522, 16.6891975
1: -69.7127380, 20.7716827, -61.8059921, 18.7006969, -88.4134293, 82.5776596
2: -35.4181366, 19.7648678, -31.7046204, 17.7976303, -53.2157669, 51.4694901
3: -47.1776009, 16.1546783, -41.9235001, 14.5004988, -61.6781006, 58.0781784
4: -24.2655697, 16.8753262, -21.8227959, 15.2470884, -39.5126572, 38.6981201

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9508501, upper bound: 18.9510806
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9508501, upper bound: 18.9510821
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -9.2812138, 10.1229925, -8.4722681, 9.1727180, -18.4539318, 18.5952587
1: -76.4536362, 23.0601673, -69.4441910, 20.9841156, -97.4377289, 92.5043488
2: -38.9970322, 21.8934002, -35.5888863, 19.9144402, -58.9114647, 57.4822769
3: -51.8654976, 17.8921623, -47.1300621, 16.2118225, -68.0773163, 65.0222244
4: -26.9062939, 18.7320652, -24.5156803, 16.9938984, -43.9001846, 43.2477341

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9509585, upper bound: 18.9511183
time: 1.11 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9509585, upper bound: 18.9511183
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -8.6733189, 9.4518604, -8.4722681, 9.1727180, -17.8460369, 17.9241276
1: -71.7629929, 21.5673923, -69.4441910, 20.9841156, -92.7471085, 91.0115738
2: -36.6312981, 20.4828300, -35.5888863, 19.9144402, -56.5457306, 56.0717163
3: -48.6514740, 16.7287254, -47.1300621, 16.2118225, -64.8632889, 63.8587837
4: -25.1728363, 17.4970837, -24.5156803, 16.9938984, -42.1667290, 42.0127563

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9509585, upper bound: 18.9511183
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9509585, upper bound: 18.9511183
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7.0502601, 7.5303478, -7.6427684, 8.2038708, -15.2541313, 15.1731167
1: -57.9750938, 17.4195271, -62.1784248, 18.8399658, -76.8150635, 79.5979385
2: -29.8670139, 16.4780712, -32.0145226, 17.8973465, -47.7643585, 48.4925919
3: -39.4090233, 13.3231850, -42.2433624, 14.5147905, -53.9238129, 55.5665436
4: -20.4826241, 13.8728504, -22.0658817, 15.2416544, -35.7242737, 35.9387283

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510164, upper bound: 18.9510416
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510164, upper bound: 18.9510416
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8.3342848, 9.0134306, -7.7623205, 8.3840523, -16.7183342, 16.7757511
1: -69.4039841, 20.6650143, -64.0669556, 19.1848564, -88.5888367, 84.7319717
2: -35.3812523, 19.6434937, -32.7754211, 18.2710876, -53.6523361, 52.4189110
3: -47.0469360, 15.9243670, -43.4367752, 14.8209906, -61.8679199, 59.3611412
4: -24.2712078, 16.6210632, -22.5004425, 15.5391674, -39.8103714, 39.1215057

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510297, upper bound: 18.9511307
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510297, upper bound: 18.9511307
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7.2846899, 7.7681618, -8.4607010, 9.0892296, -16.3739204, 16.2288609
1: -59.5617943, 17.9796467, -69.0516891, 20.8816280, -80.4434204, 87.0313339
2: -30.7534275, 16.9942493, -35.5170021, 19.7998161, -50.5532417, 52.5112495
3: -40.5216827, 13.7369814, -46.9273071, 16.0455303, -56.5672150, 60.6642838
4: -21.1304703, 14.3209028, -24.4697781, 16.8171577, -37.9476280, 38.7906761

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510196, upper bound: 18.9509639
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9510196, upper bound: 18.9509639
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.6789160, 9.3434315, -8.6922035, 9.3934326, -18.0723495, 18.0356350
1: -71.5097809, 21.4643154, -71.9164276, 21.5280018, -93.0377808, 93.3807449
2: -36.6077347, 20.3663692, -36.7504692, 20.4471722, -57.0548935, 57.1168365
3: -48.5516586, 16.4992027, -48.7856178, 16.5818462, -65.1335068, 65.2848206
4: -25.1883583, 17.2459106, -25.2572823, 17.3370113, -42.5253677, 42.5031891

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511122, upper bound: 18.9511466
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9511122, upper bound: 18.9511466
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -10.8971062, 12.1310749, -8.2709360, 8.9620838, -19.8591881, 20.4020100
1: -91.5506363, 27.1607056, -67.1549911, 20.4726467, -112.0232849, 94.3156815
2: -46.3074951, 25.9951916, -34.4323196, 19.4425774, -65.7500763, 60.4275131
3: -61.8612213, 21.3250256, -45.6168633, 15.8441610, -77.7053757, 66.9418564
4: -31.7434750, 22.3931961, -23.9085655, 16.6659527, -48.4094238, 46.3017616

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9509036, upper bound: 18.9510759
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9509036, upper bound: 18.9510759
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -10.3952227, 11.5456877, -8.2709360, 8.9620838, -19.3573036, 19.8166237
1: -87.5750275, 25.9068527, -67.1549911, 20.4726467, -108.0476761, 93.0618439
2: -44.3564682, 24.7815495, -34.4323196, 19.4425774, -63.7990417, 59.2138672
3: -59.1668015, 20.3148861, -45.6168633, 15.8441610, -75.0109634, 65.9317322
4: -30.3118305, 21.2871590, -23.9085655, 16.6659527, -46.9777679, 45.1957169

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9509145, upper bound: 18.9510437
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9509145, upper bound: 18.9510768
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -11.2018261, 12.4151487, -9.1516590, 9.9180689, -21.1198959, 21.5668068
1: -93.2709656, 27.8667412, -74.6906586, 22.6855278, -115.9564972, 102.5574036
2: -47.3498802, 26.6207619, -38.2063103, 21.4979877, -68.8478699, 64.8270721
3: -63.1093216, 21.8203583, -50.7318001, 17.5226574, -80.6319656, 72.5521469
4: -32.5403671, 22.9329338, -26.4997253, 18.3606358, -50.9010010, 49.4326515

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512311, upper bound: 18.9510679
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512167, upper bound: 18.9511078
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -10.7272062, 11.8678532, -9.1516590, 9.9180689, -20.6452713, 21.0195045
1: -89.5768433, 26.6958580, -74.6906586, 22.6855278, -112.2623672, 101.3865204
2: -45.5490761, 25.4855118, -38.2063103, 21.4979877, -67.0470657, 63.6918221
3: -60.6049881, 20.8796730, -50.7318001, 17.5226574, -78.1276321, 71.6114655
4: -31.2123833, 21.8991203, -26.4997253, 18.3606358, -49.5730209, 48.3988304

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512311, upper bound: 18.9510679
time: 1.34 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512167, upper bound: 18.9511078
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -10.4860487, 11.6041479, -8.5377665, 9.2160435, -19.7020912, 20.1419144
1: -89.0765076, 26.0819302, -69.9912872, 21.1165276, -110.1930313, 96.0731964
2: -44.8504677, 24.9929676, -35.7946892, 20.0698662, -64.9203339, 60.7876472
3: -60.1386566, 20.3818378, -47.5095062, 16.2787514, -76.4174042, 67.8913422
4: -30.6778603, 21.3189163, -24.7651558, 17.0837040, -47.7615547, 46.0840721

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9509590, upper bound: 18.9510527
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9509591, upper bound: 18.9510702
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -9.9340019, 10.9725389, -8.5377665, 9.2160435, -19.1500397, 19.5103054
1: -84.7262039, 24.7275600, -69.9912872, 21.1165276, -105.8427124, 94.7188339
2: -42.7198639, 23.6824989, -35.7946892, 20.0698662, -62.7897301, 59.4771767
3: -57.1951981, 19.2917747, -47.5095062, 16.2787514, -73.4739380, 66.8012848
4: -29.1180782, 20.1374397, -24.7651558, 17.0837040, -46.2017746, 44.9025955

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9509590, upper bound: 18.9510527
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9509591, upper bound: 18.9510702
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -10.7846775, 11.8848133, -9.4104328, 10.1656132, -20.9502907, 21.2952423
1: -90.7803650, 26.7814102, -77.4555283, 23.3104992, -114.0908661, 104.2369385
2: -45.8776779, 25.6127319, -39.5186195, 22.1127834, -67.9904633, 65.1313477
3: -61.3727188, 20.8705025, -52.5757561, 17.9455833, -79.3182983, 73.4462585
4: -31.4687290, 21.8509407, -27.3382473, 18.7696209, -50.2383499, 49.1891823

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512175, upper bound: 18.9510650
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9509591, upper bound: 18.9510669
time: 0.94 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -10.2558784, 11.2893219, -9.4104328, 10.1656132, -20.4214916, 20.6997547
1: -86.7035675, 25.4960804, -77.4555283, 23.3104992, -110.0140686, 102.9516068
2: -43.8734894, 24.3727894, -39.5186195, 22.1127834, -65.9862747, 63.8914108
3: -58.6036911, 19.8451500, -52.5757561, 17.9455833, -76.5492706, 72.4209061
4: -29.9915600, 20.7381935, -27.3382473, 18.7696209, -48.7611809, 48.0764389

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512175, upper bound: 18.9510650
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9512126, upper bound: 18.9510669
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -10.8971062, 12.1310749, -7.8220558, 8.4642687, -19.3613739, 19.9531307
1: -91.5506363, 27.1607056, -63.4693909, 19.3185883, -110.8692093, 90.6300964
2: -46.3074951, 25.9951916, -32.6523743, 18.3615723, -64.6690598, 58.6475563
3: -61.8612213, 21.3250256, -43.0904961, 14.9659500, -76.8271713, 64.4155121
4: -31.7434750, 22.3931961, -22.5190487, 15.7537756, -47.4972496, 44.9122391

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9508244, upper bound: 18.9510437
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -18.9508244, upper bound: 18.9510800
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -10.3952227, 11.5456877, -7.8220558, 8.4642687, -18.8594913, 19.3677444
1: -87.5750275, 25.9068527, -63.4693909, 19.3185883, -106.8936005, 89.3762436
2: -44.3564682, 24.7815495, -32.6523743, 18.3615723, -62.7180405, 57.4339218
3: -59.1668015, 20.3148861, -43.0904961, 14.9659500, -74.1327515, 63.4053802
4: -30.3118305, 21.2871590, -22.5190487, 15.7537756, -46.0655975, 43.8061905

Time for backsubstitution: 1.40 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.15 + 417.37 = 420.52 seconds
