## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 3897.0783163271253


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1645.1527100, 2891.7785645, -1645.1527100, 2891.7785645, -4536.9311523, 4536.9311523)
1: (-287.8174744, 365.9089050, -287.8174744, 365.9089050, -653.7263794, 653.7263794)
2: (-219.8338470, 493.6749573, -219.8338470, 493.6749573, -713.5087891, 713.5087891)
3: (-230.2796783, 647.6876831, -230.2796783, 647.6876831, -877.9673462, 877.9673462)
4: (-189.1308136, 619.7971191, -189.1308136, 619.7971191, -808.9279175, 808.9279175)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.30 + 1.98 = 4.28 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -3897.1172875, upper bound: 3897.1172875

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171606, upper bound: 3897.1172206
time: 0.66 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171587, upper bound: 3897.1171586
time: 0.65 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.51 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.51
Output dim: 0, lower bound: -3897.1171606, upper bound: 3897.1172206
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.51
Output dim: 0, lower bound: -3897.1171587, upper bound: 3897.1171586

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -1592.0351562, 2796.3410645, -1628.6346436, 2861.8005371, -4453.8344727, 4424.9746094
1: -278.7979736, 354.1008606, -284.9953308, 362.2212219, -641.0191040, 639.0961914
2: -212.6966705, 478.1099548, -217.5989075, 488.8296204, -701.5263062, 695.7088623
3: -223.1169739, 626.8599243, -228.0446320, 641.1971436, -864.3140869, 854.9044800
4: -183.0593109, 600.2265625, -187.2327423, 613.7031860, -796.7625122, 787.4592896

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171587, upper bound: 3897.1171587
time: 0.68 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171587, upper bound: 3897.1171587
time: 0.74 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -1745.4454346, 3066.9980469, -1532.2233887, 2700.3027344, -4445.7480469, 4599.2216797
1: -305.1494751, 388.8529663, -268.8605652, 341.4550476, -646.6044922, 657.7134399
2: -233.2548218, 521.6334839, -205.2992554, 458.9773865, -692.2321777, 726.9327393
3: -244.7914886, 687.5227051, -215.1436462, 603.5111694, -848.3026733, 902.6662598
4: -200.5502625, 655.0048828, -176.6123047, 576.1626587, -776.7128906, 831.6171265

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171587, upper bound: 3897.1171585
time: 0.68 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171587, upper bound: 3897.1171587
time: 0.72 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.72 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.72
Output dim: 0, lower bound: -3897.1171587, upper bound: 3897.1171587
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.72
Output dim: 0, lower bound: -3897.1171587, upper bound: 3897.1171587
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.72
Output dim: 0, lower bound: -3897.1171587, upper bound: 3897.1171585
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.72
Output dim: 0, lower bound: -3897.1171587, upper bound: 3897.1171587

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -1592.0351562, 2796.3410645, -1592.0351562, 2796.3410645, -4388.3759766, 4388.3759766
1: -278.7979736, 354.1008606, -278.7979736, 354.1008606, -632.8988037, 632.8988037
2: -212.6966705, 478.1099548, -212.6966705, 478.1099548, -690.8066406, 690.8066406
3: -223.1169739, 626.8599243, -223.1169739, 626.8599243, -849.9768677, 849.9768677
4: -183.0593109, 600.2265625, -183.0593109, 600.2265625, -783.2858887, 783.2858887

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171260, upper bound: 3897.1171193
time: 0.61 seconds

## Relational analysis of NS_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168461, upper bound: 3897.1170777
time: 0.64 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -1592.0351562, 2796.3410645, -1745.4454346, 3066.9980469, -4659.0327148, 4541.7866211
1: -278.7979736, 354.1008606, -305.1494751, 388.8529663, -667.6508789, 659.2503662
2: -212.6966705, 478.1099548, -233.2548218, 521.6334839, -734.3301392, 711.3647461
3: -223.1169739, 626.8599243, -244.7914886, 687.5227051, -910.6396484, 871.6514282
4: -183.0593109, 600.2265625, -200.5502625, 655.0048828, -838.0642090, 800.7768555

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171260, upper bound: 3897.1171194
time: 0.80 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168461, upper bound: 3897.1170777
time: 0.71 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -1745.4454346, 3066.9980469, -1592.0351562, 2796.3410645, -4541.7866211, 4659.0322266
1: -305.1494751, 388.8529663, -278.7979736, 354.1008606, -659.2503662, 667.6508789
2: -233.2548218, 521.6334839, -212.6966705, 478.1099548, -711.3647461, 734.3301392
3: -244.7914886, 687.5227051, -223.1169739, 626.8599243, -871.6514282, 910.6396484
4: -200.5502625, 655.0048828, -183.0593109, 600.2265625, -800.7768555, 838.0642090

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168565, upper bound: 3897.1171014
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168233, upper bound: 3897.1168233
time: 0.70 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -1745.4454346, 3066.9980469, -1745.4454346, 3066.9980469, -4812.4433594, 4812.4433594
1: -305.1494751, 388.8529663, -305.1494751, 388.8529663, -694.0024414, 694.0024414
2: -233.2548218, 521.6334839, -233.2548218, 521.6334839, -754.8883057, 754.8883057
3: -244.7914886, 687.5227051, -244.7914886, 687.5227051, -932.3142090, 932.3142090
4: -200.5502625, 655.0048828, -200.5502625, 655.0048828, -855.5551147, 855.5551758

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168565, upper bound: 3897.1171015
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168233, upper bound: 3897.1168233
time: 0.68 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.74 seconds
NS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 0, lower bound: -3897.1171260, upper bound: 3897.1171193
NS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 0, lower bound: -3897.1168461, upper bound: 3897.1170777
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 0, lower bound: -3897.1171260, upper bound: 3897.1171194
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 0, lower bound: -3897.1168461, upper bound: 3897.1170777
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 0, lower bound: -3897.1168565, upper bound: 3897.1171014
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 0, lower bound: -3897.1168233, upper bound: 3897.1168233
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 0, lower bound: -3897.1168565, upper bound: 3897.1171015
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 0, lower bound: -3897.1168233, upper bound: 3897.1168233

## BFS NS instance: NS_A1_B1_B1

### Backsubstitution after applying NS history:
0: -1533.6861572, 2699.0637207, -1458.6683350, 2572.2277832, -4105.9140625, 4157.7314453
1: -269.3697815, 341.5298157, -257.0128784, 325.0759583, -594.4457397, 598.5427246
2: -205.2158661, 460.7400208, -195.5194244, 438.5618286, -643.7775879, 656.2594604
3: -215.1105194, 604.3367310, -204.6784973, 574.9701538, -790.0806274, 809.0151367
4: -176.5976868, 578.4238892, -168.2840118, 550.4185791, -727.0162354, 746.7077637

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1172475, upper bound: 3897.1172475
time: 0.73 seconds

## Relational analysis of NS_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1172475, upper bound: 3897.1172666
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_B2

### Backsubstitution after applying NS history:
0: -1561.3371582, 2743.4260254, -1510.9511719, 2656.6083984, -4217.9453125, 4254.3769531
1: -273.5661316, 347.3095398, -264.9468079, 336.1708374, -609.7369385, 612.2562866
2: -208.6143188, 469.0863037, -201.9160614, 454.2324829, -662.8466187, 671.0023804
3: -218.9128418, 614.9766235, -212.0039825, 595.4115601, -814.3243408, 826.9804688
4: -179.5636139, 588.9244995, -173.8208313, 570.3193970, -749.8828735, 762.7453003

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1172666, upper bound: 3897.1172472
time: 0.75 seconds

## Relational analysis of NS_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1172666, upper bound: 3897.1172673
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -1533.6861572, 2699.0637207, -1639.6286621, 2891.3327637, -4425.0185547, 4338.6914062
1: -269.3697815, 341.5298157, -287.9586487, 366.1566162, -635.5263672, 629.4884033
2: -205.2158661, 460.7400208, -219.6820526, 490.6211243, -695.8369141, 680.4220581
3: -215.1105194, 604.3367310, -230.0631561, 646.9559326, -862.0663452, 834.3997803
4: -176.5976868, 578.4238892, -188.8361053, 615.9566040, -792.5543213, 767.2598877

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168461, upper bound: 3897.1170776
time: 0.67 seconds

## Relational analysis of NS_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168461, upper bound: 3897.1170776
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -1561.3371582, 2743.4260254, -1665.7705078, 2931.1472168, -4492.4843750, 4409.1962891
1: -273.5661316, 347.3095398, -291.7715454, 371.3513489, -644.9173584, 639.0809326
2: -208.6143188, 469.0863037, -222.7444611, 498.4445190, -707.0587158, 691.8307495
3: -218.9128418, 614.9766235, -233.8927002, 656.7106323, -875.6234131, 848.8693237
4: -179.5636139, 588.9244995, -191.5276184, 625.9798584, -805.5434570, 780.4521484

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168461, upper bound: 3897.1170777
time: 0.66 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168461, upper bound: 3897.1170777
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1639.6286621, 2891.3327637, -1533.6861572, 2699.0637207, -4338.6923828, 4425.0185547
1: -287.9586487, 366.1566162, -269.3697815, 341.5298157, -629.4884644, 635.5263672
2: -219.6820526, 490.6211243, -205.2158661, 460.7400208, -680.4220581, 695.8368530
3: -230.0631561, 646.9559326, -215.1105194, 604.3367310, -834.3997803, 862.0663452
4: -188.8361053, 615.9566040, -176.5976868, 578.4238892, -767.2598877, 792.5543213

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170776, upper bound: 3897.1168461
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170777, upper bound: 3897.1168461
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1665.7705078, 2931.1472168, -1561.3371582, 2743.4260254, -4409.1962891, 4492.4843750
1: -291.7715454, 371.3513489, -273.5661316, 347.3095398, -639.0809326, 644.9173584
2: -222.7444611, 498.4445190, -208.6143188, 469.0863037, -691.8307495, 707.0587158
3: -233.8927002, 656.7106323, -218.9128418, 614.9766235, -848.8692627, 875.6234131
4: -191.5276184, 625.9798584, -179.5636139, 588.9244995, -780.4520874, 805.5434570

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170777, upper bound: 3897.1168461
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170776, upper bound: 3897.1168461
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1639.6286621, 2891.3327637, -1687.3360596, 2969.4091797, -4609.0375977, 4578.6689453
1: -287.9586487, 366.1566162, -295.6912537, 376.3566589, -664.3152466, 661.8477783
2: -219.6820526, 490.6211243, -225.8208771, 504.4299927, -724.1120605, 716.4420166
3: -230.0631561, 646.9559326, -236.8520813, 665.1807861, -895.2439575, 883.8079224
4: -188.8361053, 615.9566040, -194.1256256, 633.3875122, -822.2235718, 810.0821533

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168233, upper bound: 3897.1168229
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168233, upper bound: 3897.1168233
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1665.7705078, 2931.1472168, -1715.1794434, 3015.5646973, -4681.3349609, 4646.3266602
1: -291.7715454, 371.3513489, -300.0787659, 382.2179260, -673.9893799, 671.4299927
2: -222.7444611, 498.4445190, -229.2703552, 512.8422241, -735.5866089, 727.7148438
3: -233.8927002, 656.7106323, -240.6553802, 675.8240356, -909.7167358, 897.3660278
4: -191.5276184, 625.9798584, -197.1297302, 644.0076294, -835.5352783, 823.1096191

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168233, upper bound: 3897.1168231
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168233, upper bound: 3897.1168233
time: 0.71 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.80 seconds
NS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.80
Output dim: 0, lower bound: -3897.1172475, upper bound: 3897.1172475
NS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.80
Output dim: 0, lower bound: -3897.1172475, upper bound: 3897.1172666
NS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.80
Output dim: 0, lower bound: -3897.1172666, upper bound: 3897.1172472
NS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.80
Output dim: 0, lower bound: -3897.1172666, upper bound: 3897.1172673
NS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.80
Output dim: 0, lower bound: -3897.1168461, upper bound: 3897.1170776
NS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.80
Output dim: 0, lower bound: -3897.1168461, upper bound: 3897.1170776
NS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.80
Output dim: 0, lower bound: -3897.1168461, upper bound: 3897.1170777
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.80
Output dim: 0, lower bound: -3897.1168461, upper bound: 3897.1170777
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.80
Output dim: 0, lower bound: -3897.1170776, upper bound: 3897.1168461
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.80
Output dim: 0, lower bound: -3897.1170777, upper bound: 3897.1168461
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.80
Output dim: 0, lower bound: -3897.1170777, upper bound: 3897.1168461
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.80
Output dim: 0, lower bound: -3897.1170776, upper bound: 3897.1168461
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.80
Output dim: 0, lower bound: -3897.1168233, upper bound: 3897.1168229
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.80
Output dim: 0, lower bound: -3897.1168233, upper bound: 3897.1168233
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.80
Output dim: 0, lower bound: -3897.1168233, upper bound: 3897.1168231
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.80
Output dim: 0, lower bound: -3897.1168233, upper bound: 3897.1168233

## BFS NS instance: NS_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -1458.6683350, 2572.2277832, -1458.6683350, 2572.2277832, -4030.8959961, 4030.8959961
1: -257.0128784, 325.0759583, -257.0128784, 325.0759583, -582.0888672, 582.0888672
2: -195.5194244, 438.5618286, -195.5194244, 438.5618286, -634.0812378, 634.0812378
3: -204.6784973, 574.9701538, -204.6784973, 574.9701538, -779.6486816, 779.6486816
4: -168.2840118, 550.4185791, -168.2840118, 550.4185791, -718.7025146, 718.7025146

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_B1_A1_A1

### Relational analysis result of NS_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1172460, upper bound: 3897.1172287
time: 0.76 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2

### Relational analysis result of NS_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1172468, upper bound: 3897.1172468
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -1510.9511719, 2656.6083984, -1458.6683350, 2572.2277832, -4083.1789551, 4115.2768555
1: -264.9468079, 336.1708374, -257.0128784, 325.0759583, -590.0227051, 593.1837158
2: -201.9160614, 454.2324829, -195.5194244, 438.5618286, -640.4777832, 649.7518921
3: -212.0039825, 595.4115601, -204.6784973, 574.9701538, -786.9741211, 800.0900879
4: -173.8208313, 570.3193970, -168.2840118, 550.4185791, -724.2393799, 738.6033325

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_B1_A2_A1

### Relational analysis result of NS_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1172460, upper bound: 3897.1172287
time: 0.61 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2

### Relational analysis result of NS_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1172468, upper bound: 3897.1172642
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -1458.6683350, 2572.2277832, -1510.9511719, 2656.6083984, -4115.2768555, 4083.1789551
1: -257.0128784, 325.0759583, -264.9468079, 336.1708374, -593.1837158, 590.0227051
2: -195.5194244, 438.5618286, -201.9160614, 454.2324829, -649.7518921, 640.4777222
3: -204.6784973, 574.9701538, -212.0039825, 595.4115601, -800.0900879, 786.9741211
4: -168.2840118, 550.4185791, -173.8208313, 570.3193970, -738.6033325, 724.2393799

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_B2_A1_B1

### Relational analysis result of NS_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1172287, upper bound: 3897.1172458
time: 0.68 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2

### Relational analysis result of NS_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1172468, upper bound: 3897.1172467
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -1510.9511719, 2656.6083984, -1510.9511719, 2656.6083984, -4167.5595703, 4167.5595703
1: -264.9468079, 336.1708374, -264.9468079, 336.1708374, -601.1176758, 601.1176758
2: -201.9160614, 454.2324829, -201.9160614, 454.2324829, -656.1484985, 656.1484985
3: -212.0039825, 595.4115601, -212.0039825, 595.4115601, -807.4154663, 807.4154663
4: -173.8208313, 570.3193970, -173.8208313, 570.3193970, -744.1401367, 744.1401367

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_B2_A2_A1

### Relational analysis result of NS_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1172460, upper bound: 3897.1172287
time: 0.66 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2

### Relational analysis result of NS_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1172468, upper bound: 3897.1172642
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -1458.6683350, 2572.2277832, -1639.6286621, 2891.3327637, -4350.0009766, 4211.8564453
1: -257.0128784, 325.0759583, -287.9586487, 366.1566162, -623.1694336, 613.0345459
2: -195.5194244, 438.5618286, -219.6820526, 490.6211243, -686.1405640, 658.2438965
3: -204.6784973, 574.9701538, -230.0631561, 646.9559326, -851.6343994, 805.0333252
4: -168.2840118, 550.4185791, -188.8361053, 615.9566040, -784.2405396, 739.2546387

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_B1_A1_B1

### Relational analysis result of NS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170381, upper bound: 3897.1168977
time: 0.74 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2

### Relational analysis result of NS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170381, upper bound: 3897.1170960
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -1510.9511719, 2656.6083984, -1639.6286621, 2891.3327637, -4402.2836914, 4296.2368164
1: -264.9468079, 336.1708374, -287.9586487, 366.1566162, -631.1033325, 624.1295166
2: -201.9160614, 454.2324829, -219.6820526, 490.6211243, -692.5371094, 673.9145508
3: -212.0039825, 595.4115601, -230.0631561, 646.9559326, -858.9598389, 825.4747314
4: -173.8208313, 570.3193970, -188.8361053, 615.9566040, -789.7774048, 759.1554565

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B1_A2_B1

### Relational analysis result of NS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171198, upper bound: 3897.1170919
time: 0.69 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2

### Relational analysis result of NS_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171260, upper bound: 3897.1171194
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -1458.6683350, 2572.2277832, -1665.7705078, 2931.1472168, -4389.8154297, 4237.9980469
1: -257.0128784, 325.0759583, -291.7715454, 371.3513489, -628.3641968, 616.8473511
2: -195.5194244, 438.5618286, -222.7444611, 498.4445190, -693.9639282, 661.3062134
3: -204.6784973, 574.9701538, -233.8927002, 656.7106323, -861.3890991, 808.8628540
4: -168.2840118, 550.4185791, -191.5276184, 625.9798584, -794.2638550, 741.9461670

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_B2_A1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1164013, upper bound: 3897.1165872
time: 0.60 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2

### Relational analysis result of NS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168387, upper bound: 3897.1170669
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -1510.9511719, 2656.6083984, -1665.7705078, 2931.1472168, -4442.0986328, 4322.3779297
1: -264.9468079, 336.1708374, -291.7715454, 371.3513489, -636.2980347, 627.9423218
2: -201.9160614, 454.2324829, -222.7444611, 498.4445190, -700.3605957, 676.9769287
3: -212.0039825, 595.4115601, -233.8927002, 656.7106323, -868.7145386, 829.3042603
4: -173.8208313, 570.3193970, -191.5276184, 625.9798584, -799.8006592, 761.8469849

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_B2_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1164013, upper bound: 3897.1165872
time: 0.59 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2

### Relational analysis result of NS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168387, upper bound: 3897.1170667
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1639.6286621, 2891.3327637, -1458.6683350, 2572.2277832, -4211.8564453, 4350.0009766
1: -287.9586487, 366.1566162, -257.0128784, 325.0759583, -613.0345459, 623.1694336
2: -219.6820526, 490.6211243, -195.5194244, 438.5618286, -658.2438965, 686.1405640
3: -230.0631561, 646.9559326, -204.6784973, 574.9701538, -805.0333252, 851.6343994
4: -188.8361053, 615.9566040, -168.2840118, 550.4185791, -739.2546387, 784.2405396

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168977, upper bound: 3897.1170381
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170960, upper bound: 3897.1171179
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1639.6286621, 2891.3327637, -1510.9511719, 2656.6083984, -4296.2368164, 4402.2836914
1: -287.9586487, 366.1566162, -264.9468079, 336.1708374, -624.1295166, 631.1033325
2: -219.6820526, 490.6211243, -201.9160614, 454.2324829, -673.9145508, 692.5370483
3: -230.0631561, 646.9559326, -212.0039825, 595.4115601, -825.4747314, 858.9597778
4: -188.8361053, 615.9566040, -173.8208313, 570.3193970, -759.1553955, 789.7774048

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170919, upper bound: 3897.1171198
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171194, upper bound: 3897.1171260
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1665.7705078, 2931.1472168, -1458.6683350, 2572.2277832, -4237.9980469, 4389.8154297
1: -291.7715454, 371.3513489, -257.0128784, 325.0759583, -616.8473511, 628.3641357
2: -222.7444611, 498.4445190, -195.5194244, 438.5618286, -661.3061523, 693.9639282
3: -233.8927002, 656.7106323, -204.6784973, 574.9701538, -808.8628540, 861.3890991
4: -191.5276184, 625.9798584, -168.2840118, 550.4185791, -741.9461670, 794.2638550

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1165877, upper bound: 3897.1164010
time: 0.55 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170669, upper bound: 3897.1168389
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1665.7705078, 2931.1472168, -1510.9511719, 2656.6083984, -4322.3779297, 4442.0986328
1: -291.7715454, 371.3513489, -264.9468079, 336.1708374, -627.9423218, 636.2980347
2: -222.7444611, 498.4445190, -201.9160614, 454.2324829, -676.9769287, 700.3605957
3: -233.8927002, 656.7106323, -212.0039825, 595.4115601, -829.3042603, 868.7145386
4: -191.5276184, 625.9798584, -173.8208313, 570.3193970, -761.8470459, 799.8006592

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1165877, upper bound: 3897.1164013
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170669, upper bound: 3897.1168389
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1639.6286621, 2891.3327637, -1639.6286621, 2891.3327637, -4530.9614258, 4530.9614258
1: -287.9586487, 366.1566162, -287.9586487, 366.1566162, -654.1151123, 654.1151123
2: -219.6820526, 490.6211243, -219.6820526, 490.6211243, -710.3031616, 710.3031616
3: -230.0631561, 646.9559326, -230.0631561, 646.9559326, -877.0190430, 877.0191040
4: -188.8361053, 615.9566040, -188.8361053, 615.9566040, -804.7926025, 804.7926025

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168565, upper bound: 3897.1170945
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168553, upper bound: 3897.1171016
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1639.6286621, 2891.3327637, -1665.7705078, 2931.1472168, -4570.7758789, 4557.1025391
1: -287.9586487, 366.1566162, -291.7715454, 371.3513489, -659.3098755, 657.9279175
2: -219.6820526, 490.6211243, -222.7444611, 498.4445190, -718.1265869, 713.3655396
3: -230.0631561, 646.9559326, -233.8927002, 656.7106323, -886.7737427, 880.8486328
4: -188.8361053, 615.9566040, -191.5276184, 625.9798584, -814.8159790, 807.4841919

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1167514, upper bound: 3897.1170486
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2

### Relational analysis result of NS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168553, upper bound: 3897.1171015
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1665.7705078, 2931.1472168, -1639.6286621, 2891.3327637, -4557.1025391, 4570.7758789
1: -291.7715454, 371.3513489, -287.9586487, 366.1566162, -657.9279175, 659.3098755
2: -222.7444611, 498.4445190, -219.6820526, 490.6211243, -713.3654785, 718.1265869
3: -233.8927002, 656.7106323, -230.0631561, 646.9559326, -880.8485718, 886.7737427
4: -191.5276184, 625.9798584, -188.8361053, 615.9566040, -807.4841919, 814.8159790

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1167682, upper bound: 3897.1167130
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168233, upper bound: 3897.1168233
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1665.7705078, 2931.1472168, -1665.7705078, 2931.1472168, -4596.9174805, 4596.9174805
1: -291.7715454, 371.3513489, -291.7715454, 371.3513489, -663.1226807, 663.1226807
2: -222.7444611, 498.4445190, -222.7444611, 498.4445190, -721.1889648, 721.1889648
3: -233.8927002, 656.7106323, -233.8927002, 656.7106323, -890.6033325, 890.6033325
4: -191.5276184, 625.9798584, -191.5276184, 625.9798584, -817.5074463, 817.5074463

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1167682, upper bound: 3897.1167128
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168231, upper bound: 3897.1168231
time: 0.67 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.22 seconds
NS_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -3897.1172460, upper bound: 3897.1172287
NS_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -3897.1172468, upper bound: 3897.1172468
NS_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -3897.1172460, upper bound: 3897.1172287
NS_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -3897.1172468, upper bound: 3897.1172642
NS_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -3897.1172287, upper bound: 3897.1172458
NS_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -3897.1172468, upper bound: 3897.1172467
NS_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -3897.1172460, upper bound: 3897.1172287
NS_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -3897.1172468, upper bound: 3897.1172642
NS_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -3897.1170381, upper bound: 3897.1168977
NS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -3897.1170381, upper bound: 3897.1170960
NS_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -3897.1171198, upper bound: 3897.1170919
NS_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -3897.1171260, upper bound: 3897.1171194
NS_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -3897.1164013, upper bound: 3897.1165872
NS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -3897.1168387, upper bound: 3897.1170669
NS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -3897.1164013, upper bound: 3897.1165872
NS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -3897.1168387, upper bound: 3897.1170667
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -3897.1168977, upper bound: 3897.1170381
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -3897.1170960, upper bound: 3897.1171179
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -3897.1170919, upper bound: 3897.1171198
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -3897.1171194, upper bound: 3897.1171260
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -3897.1165877, upper bound: 3897.1164010
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -3897.1170669, upper bound: 3897.1168389
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -3897.1165877, upper bound: 3897.1164013
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -3897.1170669, upper bound: 3897.1168389
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -3897.1168565, upper bound: 3897.1170945
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -3897.1168553, upper bound: 3897.1171016
NS_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -3897.1167514, upper bound: 3897.1170486
NS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -3897.1168553, upper bound: 3897.1171015
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -3897.1167682, upper bound: 3897.1167130
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -3897.1168233, upper bound: 3897.1168233
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -3897.1167682, upper bound: 3897.1167128
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 0, lower bound: -3897.1168231, upper bound: 3897.1168231

## BFS NS instance: NS_A1_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -1417.8165283, 2502.5986328, -1428.5247803, 2519.7097168, -3937.5258789, 3931.1235352
1: -250.3135223, 316.1636353, -251.7532349, 318.2980957, -568.6115723, 567.9167480
2: -190.3193970, 427.1272278, -191.5401306, 429.5758362, -619.8952637, 618.6673584
3: -199.2041016, 559.2181396, -200.4884796, 563.0109863, -762.2150879, 759.7066040
4: -163.8197632, 536.0412598, -164.8744049, 539.1256104, -702.9453125, 700.9154663

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_B1_A1_A1_B1

### Relational analysis result of NS_A1_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1172299, upper bound: 3897.1172299
time: 0.81 seconds

## Relational analysis of NS_A1_B1_B1_A1_A1_B2

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1172298, upper bound: 3897.1172299
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -1428.6033936, 2519.1269531, -1440.5523682, 2540.3820801, -3968.9853516, 3959.6791992
1: -251.7466888, 318.2354431, -253.8581238, 320.9662170, -572.7128906, 572.0935059
2: -191.4405518, 429.7970581, -193.0669861, 433.3073120, -624.7478638, 622.8638916
3: -200.3177490, 563.2192383, -202.0449219, 567.9108887, -768.2286377, 765.2641602
4: -164.7749176, 539.3828125, -166.1717987, 543.8009644, -708.5758667, 705.5546265

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_B1_A1_A2_B1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1172299, upper bound: 3897.1172467
time: 0.70 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1172299, upper bound: 3897.1172470
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -1476.8770752, 2599.3308105, -1428.5247803, 2519.7097168, -3996.5866699, 4027.8554688
1: -259.4418030, 328.6999207, -251.7532349, 318.2980957, -577.7398682, 580.4531250
2: -197.5400696, 444.9430542, -191.5401306, 429.5758362, -627.1159058, 636.4831543
3: -207.2708130, 582.3969116, -200.4884796, 563.0109863, -770.2817993, 782.8853760
4: -170.0515594, 558.5493164, -164.8744049, 539.1256104, -709.1771240, 723.4235840

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_B1_A2_A1_B1

### Relational analysis result of NS_A1_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171981, upper bound: 3897.1170034
time: 0.69 seconds

## Relational analysis of NS_A1_B1_B1_A2_A1_B2

### Relational analysis result of NS_A1_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171709, upper bound: 3897.1170020
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -1476.4704590, 2595.3466797, -1440.5523682, 2540.3820801, -4016.8520508, 4035.8989258
1: -258.8100281, 328.3325806, -253.8581238, 320.9662170, -579.7762451, 582.1906738
2: -197.2351990, 444.0037842, -193.0669861, 433.3073120, -630.5424805, 637.0706787
3: -207.0301361, 581.8175659, -202.0449219, 567.9108887, -774.9410400, 783.8624878
4: -169.7957153, 557.4502563, -166.1717987, 543.8009644, -713.5966187, 723.6219482

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1172297, upper bound: 3897.1172610
time: 0.70 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1172297, upper bound: 3897.1172642
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1428.5247803, 2519.7097168, -1476.8770752, 2599.3308105, -4027.8554688, 3996.5866699
1: -251.7532349, 318.2980957, -259.4418030, 328.6999207, -580.4531250, 577.7398682
2: -191.5401306, 429.5758362, -197.5400696, 444.9430542, -636.4831543, 627.1159058
3: -200.4884796, 563.0109863, -207.2708130, 582.3969116, -782.8853760, 770.2817993
4: -164.8744049, 539.1256104, -170.0515594, 558.5493164, -723.4235840, 709.1771851

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170034, upper bound: 3897.1171979
time: 0.68 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170020, upper bound: 3897.1171706
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1440.5523682, 2540.3820801, -1476.4704590, 2595.3466797, -4035.8989258, 4016.8520508
1: -253.8581238, 320.9662170, -258.8100281, 328.3325806, -582.1906738, 579.7762451
2: -193.0669861, 433.3073120, -197.2351990, 444.0037842, -637.0706787, 630.5424805
3: -202.0449219, 567.9108887, -207.0301361, 581.8175659, -783.8624878, 774.9410400
4: -166.1717987, 543.8009644, -169.7957153, 557.4502563, -723.6219482, 713.5966187

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1172610, upper bound: 3897.1172297
time: 0.81 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1172610, upper bound: 3897.1172465
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -1476.8770752, 2599.3308105, -1481.6865234, 2606.0788574, -4082.9553223, 4081.0173340
1: -259.4418030, 328.6999207, -259.9767151, 329.6315613, -589.0733643, 588.6766357
2: -197.5400696, 444.9430542, -198.0433655, 445.8274231, -643.3674927, 642.9862671
3: -207.2708130, 582.3969116, -207.8647919, 584.0599365, -791.3307495, 790.2617188
4: -170.0515594, 558.5493164, -170.4842834, 559.7058716, -729.7574463, 729.0335693

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1172284, upper bound: 3897.1172284
time: 0.68 seconds

## Relational analysis of NS_A1_B1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1172284, upper bound: 3897.1172287
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -1476.4704590, 2595.3466797, -1490.9035645, 2620.9902344, -4097.4604492, 4086.2492676
1: -258.8100281, 328.3325806, -261.3812866, 331.6135864, -590.4235840, 589.7138062
2: -197.2351990, 444.0037842, -199.1943207, 448.2905579, -645.5256958, 643.1981201
3: -207.0301361, 581.8175659, -209.1106110, 587.5097046, -794.5398560, 790.9281616
4: -169.7957153, 557.4502563, -171.4798584, 562.8439941, -732.6397095, 728.9301147

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1172287, upper bound: 3897.1172611
time: 0.68 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1172287, upper bound: 3897.1172642
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1440.8282471, 2542.1904297, -1594.9783936, 2815.5861816, -4256.4145508, 4137.1684570
1: -254.1220551, 321.2551575, -280.5543518, 356.4607239, -610.5827637, 601.8093262
2: -193.2313690, 433.5679932, -213.8440857, 477.8609009, -671.0921631, 647.4120483
3: -202.1988525, 568.1939087, -223.7975922, 629.7775879, -831.9764404, 791.9913940
4: -166.3129272, 544.1195679, -183.7783203, 600.0015869, -766.3145142, 727.8978882

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B1_A1_B1_B1

### Relational analysis result of NS_A1_B2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170837, upper bound: 3897.1171325
time: 0.65 seconds

## Relational analysis of NS_A1_B2_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_B1_A1_B1_B1

### Relational analysis result of NS_A1_B2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170287, upper bound: 3897.1168891
time: 0.73 seconds

## Relational analysis of NS_A1_B2_B1_A1_B1_B2

### Relational analysis result of NS_A1_B2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168864, upper bound: 3897.1168612
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1368.9040527, 2412.8205566, -2666.4934082, 4678.7856445, -6045.0366211, 5079.3129883
1: -240.9852142, 304.9341125, -463.2645264, 593.1577759, -834.1430054, 768.1986084
2: -183.4393311, 410.5083618, -356.4089050, 800.1788330, -982.5321655, 766.9172363
3: -192.1883545, 539.0390015, -372.2882690, 1050.7088623, -1240.6651611, 911.3272705
4: -157.8574982, 515.3161011, -306.6109619, 1004.0152588, -1159.5039062, 821.9270630

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B1_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170837, upper bound: 3897.1171600
time: 0.68 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_B1_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1167369, upper bound: 3897.1170162
time: 0.76 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2_B2

### Relational analysis result of NS_A1_B2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170426, upper bound: 3897.1170596
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1481.6865234, 2606.0788574, -1574.2062988, 2778.3317871, -4260.0185547, 4180.2846680
1: -259.9767151, 329.6315613, -276.9520569, 351.5756836, -611.5523682, 606.5836182
2: -198.0433655, 445.8274231, -211.1287842, 471.7639465, -669.8071899, 656.9561768
3: -207.8647919, 584.0599365, -221.0352783, 621.2825928, -829.1473999, 805.0952148
4: -170.4842834, 559.7058716, -181.4875031, 592.2689209, -762.7531738, 741.1933594

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_B1_A2_B1_B1

### Relational analysis result of NS_A1_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170343, upper bound: 3897.1168974
time: 0.75 seconds

## Relational analysis of NS_A1_B2_B1_A2_B1_B2

### Relational analysis result of NS_A1_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171071, upper bound: 3897.1170472
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1490.9035645, 2620.9902344, -1617.9702148, 2852.7858887, -4343.6894531, 4238.9604492
1: -261.3812866, 331.6135864, -284.1601868, 361.2568970, -622.6381836, 615.7737427
2: -199.1943207, 448.2905579, -216.7185364, 484.5160217, -683.7103271, 665.0089111
3: -209.1106110, 587.5097046, -226.9315186, 638.5805664, -847.6911621, 814.4412231
4: -171.4798584, 562.8439941, -186.2865601, 608.2865601, -779.7664185, 749.1305542

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170880, upper bound: 3897.1169324
time: 0.67 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170880, upper bound: 3897.1171194
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1440.8282471, 2542.1904297, -1622.2442627, 2858.0742188, -4298.9023438, 4164.4345703
1: -254.1220551, 321.2551575, -284.5743713, 361.9688110, -616.0908813, 605.8294678
2: -193.2313690, 433.5679932, -217.0584869, 485.9965820, -679.2279663, 650.6264648
3: -202.1988525, 568.1939087, -227.7369995, 640.0209351, -842.2197266, 795.9308472
4: -166.3129272, 544.1195679, -186.6148682, 610.3317871, -776.6447144, 730.7344360

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B2_A1_B1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1164760, upper bound: 3897.1169015
time: 0.65 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_B2_A1_B1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1129885, upper bound: 3897.1145315
time: 0.67 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_B2

### Relational analysis result of NS_A1_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1162809, upper bound: 3897.1167677
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1368.9040527, 2412.8205566, -2721.6940918, 4767.9541016, -6135.6743164, 5134.5141602
1: -240.9852142, 304.9341125, -471.5650635, 604.5075073, -845.4926758, 776.4991455
2: -183.4393311, 410.5083618, -363.1402283, 816.0269775, -998.6683960, 773.6485596
3: -192.1883545, 539.0390015, -379.9474792, 1071.4458008, -1261.8266602, 918.9863892
4: -157.8574982, 515.3161011, -312.5372009, 1023.9938965, -1179.8696289, 827.8532715

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B2_A1_B2_B1

### Relational analysis result of NS_A1_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1167505, upper bound: 3897.1170729
time: 0.67 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168542, upper bound: 3897.1171298
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1493.3034668, 2626.7263184, -1622.2442627, 2858.0742188, -4351.3779297, 4248.9707031
1: -262.0610962, 332.3471680, -284.5743713, 361.9688110, -624.0298462, 616.9215088
2: -199.6306610, 449.1902161, -217.0584869, 485.9965820, -685.6271973, 666.2487183
3: -209.5117645, 588.6362305, -227.7369995, 640.0209351, -849.5327148, 816.3731689
4: -171.8515472, 563.9599609, -186.6148682, 610.3317871, -782.1832275, 750.5748291

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B2_A2_B1_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1163944, upper bound: 3897.1165745
time: 0.57 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_B2_A2_B1_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1125724, upper bound: 3897.1125730
time: 0.76 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_B2

### Relational analysis result of NS_A1_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1161487, upper bound: 3897.1163598
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1415.9468994, 2487.5473633, -2721.6940918, 4767.9541016, -6183.9008789, 5209.2397461
1: -248.0329895, 314.7828369, -471.5650635, 604.5075073, -852.5405273, 786.3478394
2: -189.0641632, 425.0886536, -363.1402283, 816.0269775, -1005.0911255, 788.2288208
3: -198.7328949, 557.6360474, -379.9474792, 1071.4458008, -1269.8723145, 937.5834351
4: -162.7227325, 533.8887939, -312.5372009, 1023.9938965, -1186.4660645, 846.4258423

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B2_A2_B2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1167258, upper bound: 3897.1169089
time: 0.59 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_B2_A2_B2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1166163, upper bound: 3897.1167965
time: 0.73 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_B2

### Relational analysis result of NS_A1_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1166212, upper bound: 3897.1167933
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1594.9783936, 2815.5861816, -1440.8282471, 2542.1904297, -4137.1689453, 4256.4145508
1: -280.5543518, 356.4607239, -254.1220551, 321.2551575, -601.8093872, 610.5827637
2: -213.8440857, 477.8609009, -193.2313690, 433.5679932, -647.4120483, 671.0921631
3: -223.7975922, 629.7775879, -202.1988525, 568.1939087, -791.9913940, 831.9764404
4: -183.7783203, 600.0015869, -166.3129272, 544.1195679, -727.8978882, 766.3145142

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1_A1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171325, upper bound: 3897.1170837
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B1_A1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168894, upper bound: 3897.1170287
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168613, upper bound: 3897.1168864
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2666.4934082, 4678.7856445, -1368.9040527, 2412.8205566, -5079.3129883, 6045.0366211
1: -463.2645264, 593.1577759, -240.9852142, 304.9341125, -768.1986084, 834.1430054
2: -356.4089050, 800.1788330, -183.4393311, 410.5083618, -766.9172363, 982.5321655
3: -372.2882690, 1050.7088623, -192.1883545, 539.0390015, -911.3272705, 1240.6651611
4: -306.6109619, 1004.0152588, -157.8574982, 515.3161011, -821.9270630, 1159.5039062

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1_A2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171325, upper bound: 3897.1171247
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171588, upper bound: 3897.1171321
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1574.2062988, 2778.3317871, -1481.6865234, 2606.0788574, -4180.2846680, 4260.0185547
1: -276.9520569, 351.5756836, -259.9767151, 329.6315613, -606.5836182, 611.5523682
2: -211.1287842, 471.7639465, -198.0433655, 445.8274231, -656.9561768, 669.8072510
3: -221.0352783, 621.2825928, -207.8647919, 584.0599365, -805.0952148, 829.1473999
4: -181.4875031, 592.2689209, -170.4842834, 559.7058716, -741.1933594, 762.7531738

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B2_A1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168973, upper bound: 3897.1170343
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170472, upper bound: 3897.1171071
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1617.9702148, 2852.7858887, -1490.9035645, 2620.9902344, -4238.9604492, 4343.6894531
1: -284.1601868, 361.2568970, -261.3812866, 331.6135864, -615.7738037, 622.6381836
2: -216.7185364, 484.5160217, -199.1943207, 448.2905579, -665.0089111, 683.7103271
3: -226.9315186, 638.5805664, -209.1106110, 587.5097046, -814.4412231, 847.6911621
4: -186.2865601, 608.2865601, -171.4798584, 562.8439941, -749.1305542, 779.7664185

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1169324, upper bound: 3897.1170811
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1169324, upper bound: 3897.1171260
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1622.2442627, 2858.0742188, -1440.8282471, 2542.1904297, -4164.4345703, 4298.9023438
1: -284.5743713, 361.9688110, -254.1220551, 321.2551575, -605.8294067, 616.0908813
2: -217.0584869, 485.9965820, -193.2313690, 433.5679932, -650.6264648, 679.2279053
3: -227.7369995, 640.0209351, -202.1988525, 568.1939087, -795.9308472, 842.2197266
4: -186.6148682, 610.3317871, -166.3129272, 544.1195679, -730.7344360, 776.6447144

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1169015, upper bound: 3897.1164760
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1145315, upper bound: 3897.1129885
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1167668, upper bound: 3897.1162808
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2721.6940918, 4767.9541016, -1368.9040527, 2412.8205566, -5134.5141602, 6135.6743164
1: -471.5650635, 604.5075073, -240.9852142, 304.9341125, -776.4991455, 845.4926758
2: -363.1402283, 816.0269775, -183.4393311, 410.5083618, -773.6485596, 998.6683960
3: -379.9474792, 1071.4458008, -192.1883545, 539.0390015, -918.9863892, 1261.8266602
4: -312.5372009, 1023.9938965, -157.8574982, 515.3161011, -827.8532715, 1179.8696289

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170729, upper bound: 3897.1167505
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171298, upper bound: 3897.1168542
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1622.2442627, 2858.0742188, -1493.3034668, 2626.7263184, -4248.9707031, 4351.3779297
1: -284.5743713, 361.9688110, -262.0610962, 332.3471680, -616.9214478, 624.0298462
2: -217.0584869, 485.9965820, -199.6306610, 449.1902161, -666.2487183, 685.6271973
3: -227.7369995, 640.0209351, -209.5117645, 588.6362305, -816.3731689, 849.5327148
4: -186.6148682, 610.3317871, -171.8515472, 563.9599609, -750.5748291, 782.1832275

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B2_A1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1165745, upper bound: 3897.1163944
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B2_A1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1149912, upper bound: 3897.1130756
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1163605, upper bound: 3897.1161487
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2721.6940918, 4767.9541016, -1415.9468994, 2487.5473633, -5209.2397461, 6183.9008789
1: -471.5650635, 604.5075073, -248.0329895, 314.7828369, -786.3478394, 852.5404663
2: -363.1402283, 816.0269775, -189.0641632, 425.0886536, -788.2288208, 1005.0911255
3: -379.9474792, 1071.4458008, -198.7328949, 557.6360474, -937.5834351, 1269.8723145
4: -312.5372009, 1023.9938965, -162.7227325, 533.8887939, -846.4258423, 1186.4659424

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B2_A2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1169089, upper bound: 3897.1167259
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170669, upper bound: 3897.1168389
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1574.2062988, 2778.3317871, -1600.8663330, 2823.0336914, -4397.2397461, 4379.1982422
1: -276.9520569, 351.5756836, -281.1653748, 357.3133545, -634.2653809, 632.7410889
2: -211.1287842, 471.7639465, -214.4712219, 479.1498108, -690.2785034, 686.2351685
3: -221.0352783, 621.2825928, -224.5854340, 631.5429688, -852.5782471, 845.8680420
4: -181.4875031, 592.2689209, -184.3865967, 601.5070801, -782.9945679, 776.6555176

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171267, upper bound: 3897.1171267
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171267, upper bound: 3897.1171267
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1617.9702148, 2852.7858887, -1626.4765625, 2868.2468262, -4486.2167969, 4479.2626953
1: -284.1601868, 361.2568970, -285.6694946, 363.2136230, -647.3737183, 646.9263306
2: -216.7185364, 484.5160217, -217.9011688, 486.9490356, -703.6674805, 702.4171753
3: -226.9315186, 638.5805664, -228.1687164, 641.9199829, -868.8515015, 866.7492676
4: -186.2865601, 608.2865601, -187.3008423, 611.3448486, -797.6314087, 795.5874023

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171267, upper bound: 3897.1171327
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171267, upper bound: 3897.1171327
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -1600.8663330, 2823.0336914, -1615.8100586, 2845.5739746, -4446.4404297, 4438.8427734
1: -281.1653748, 357.3133545, -283.5154419, 360.3100891, -641.4754639, 640.8287964
2: -214.4712219, 479.1498108, -216.2539520, 484.5065308, -698.9777832, 695.4037476
3: -224.5854340, 631.5429688, -226.8656464, 637.4177246, -862.0031738, 858.4086304
4: -184.3865967, 601.5070801, -185.8885498, 608.3678589, -792.7543335, 787.3956299

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1167514, upper bound: 3897.1170431
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1167514, upper bound: 3897.1170490
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -1626.4765625, 2868.2468262, -1640.8540039, 2887.0419922, -4513.5185547, 4509.1005859
1: -285.6694946, 363.2136230, -287.3393555, 365.7082214, -651.3775635, 650.5529785
2: -217.9011688, 486.9490356, -219.3288574, 491.2373962, -709.1385498, 706.2778931
3: -228.1687164, 641.9199829, -230.2859344, 646.9557495, -875.1244507, 872.2059326
4: -187.3008423, 611.3448486, -188.6141968, 616.9411011, -804.2419434, 799.9590454

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168553, upper bound: 3897.1170947
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168553, upper bound: 3897.1171014
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1615.8100586, 2845.5739746, -1600.8663330, 2823.0336914, -4438.8427734, 4446.4404297
1: -283.5154419, 360.3100891, -281.1653748, 357.3133545, -640.8287964, 641.4754639
2: -216.2539520, 484.5065308, -214.4712219, 479.1498108, -695.4036865, 698.9777222
3: -226.8656464, 637.4177246, -224.5854340, 631.5429688, -858.4086304, 862.0031738
4: -185.8885498, 608.3678589, -184.3865967, 601.5070801, -787.3956299, 792.7543945

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170435, upper bound: 3897.1167514
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170435, upper bound: 3897.1167513
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1640.8540039, 2887.0419922, -1626.4765625, 2868.2468262, -4509.1005859, 4513.5185547
1: -287.3393555, 365.7082214, -285.6694946, 363.2136230, -650.5529785, 651.3775635
2: -219.3288574, 491.2373962, -217.9011688, 486.9490356, -706.2778931, 709.1385498
3: -230.2859344, 646.9557495, -228.1687164, 641.9199829, -872.2059326, 875.1244507
4: -188.6141968, 616.9411011, -187.3008423, 611.3448486, -799.9590454, 804.2419434

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170947, upper bound: 3897.1168552
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170947, upper bound: 3897.1168553
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1615.8100586, 2845.5739746, -1629.2753906, 2867.8386230, -4483.6479492, 4474.8491211
1: -283.5154419, 360.3100891, -285.5863037, 363.2123413, -646.7277222, 645.8962402
2: -216.2539520, 484.5065308, -217.9213867, 488.0002747, -704.2542114, 702.4279175
3: -226.8656464, 637.4177246, -228.6838074, 642.5330200, -869.3986816, 866.1015625
4: -185.8885498, 608.3678589, -187.3455658, 612.8074341, -798.6959839, 795.7133789

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1166729, upper bound: 3897.1166729
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1166729, upper bound: 3897.1167128
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1640.8540039, 2887.0419922, -1651.3179932, 2905.5295410, -4546.3833008, 4538.3598633
1: -287.3393555, 365.7082214, -289.1856689, 368.0728760, -655.4121094, 654.8937988
2: -219.3288574, 491.2373962, -220.7566681, 494.2677917, -713.5966797, 711.9940796
3: -230.2859344, 646.9557495, -231.7999115, 651.0517578, -881.3377075, 878.7556763
4: -188.6141968, 616.9411011, -189.8344116, 620.7412720, -809.3554688, 806.7755127

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1167130, upper bound: 3897.1167679
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1167130, upper bound: 3897.1168233
time: 0.79 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.98 seconds
NS_A1_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1172299, upper bound: 3897.1172299
NS_A1_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1172298, upper bound: 3897.1172299
NS_A1_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1172299, upper bound: 3897.1172467
NS_A1_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1172299, upper bound: 3897.1172470
NS_A1_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1171981, upper bound: 3897.1170034
NS_A1_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1171709, upper bound: 3897.1170020
NS_A1_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1172297, upper bound: 3897.1172610
NS_A1_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1172297, upper bound: 3897.1172642
NS_A1_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1170034, upper bound: 3897.1171979
NS_A1_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1170020, upper bound: 3897.1171706
NS_A1_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1172610, upper bound: 3897.1172297
NS_A1_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1172610, upper bound: 3897.1172465
NS_A1_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1172284, upper bound: 3897.1172284
NS_A1_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1172284, upper bound: 3897.1172287
NS_A1_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1172287, upper bound: 3897.1172611
NS_A1_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1172287, upper bound: 3897.1172642
NS_A1_B2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1170287, upper bound: 3897.1168891
NS_A1_B2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1168864, upper bound: 3897.1168612
NS_A1_B2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1167369, upper bound: 3897.1170162
NS_A1_B2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1170426, upper bound: 3897.1170596
NS_A1_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1170343, upper bound: 3897.1168974
NS_A1_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1171071, upper bound: 3897.1170472
NS_A1_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1170880, upper bound: 3897.1169324
NS_A1_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1170880, upper bound: 3897.1171194
NS_A1_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1129885, upper bound: 3897.1145315
NS_A1_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1162809, upper bound: 3897.1167677
NS_A1_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1167505, upper bound: 3897.1170729
NS_A1_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1168542, upper bound: 3897.1171298
NS_A1_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1125724, upper bound: 3897.1125730
NS_A1_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1161487, upper bound: 3897.1163598
NS_A1_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1166163, upper bound: 3897.1167965
NS_A1_B2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1166212, upper bound: 3897.1167933
NS_A2_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1168894, upper bound: 3897.1170287
NS_A2_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1168613, upper bound: 3897.1168864
NS_A2_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1171325, upper bound: 3897.1171247
NS_A2_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1171588, upper bound: 3897.1171321
NS_A2_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1168973, upper bound: 3897.1170343
NS_A2_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1170472, upper bound: 3897.1171071
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1169324, upper bound: 3897.1170811
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1169324, upper bound: 3897.1171260
NS_A2_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1145315, upper bound: 3897.1129885
NS_A2_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1167668, upper bound: 3897.1162808
NS_A2_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1170729, upper bound: 3897.1167505
NS_A2_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1171298, upper bound: 3897.1168542
NS_A2_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1149912, upper bound: 3897.1130756
NS_A2_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1163605, upper bound: 3897.1161487
NS_A2_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1169089, upper bound: 3897.1167259
NS_A2_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1170669, upper bound: 3897.1168389
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1171267, upper bound: 3897.1171267
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1171267, upper bound: 3897.1171267
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1171267, upper bound: 3897.1171327
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1171267, upper bound: 3897.1171327
NS_A2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1167514, upper bound: 3897.1170431
NS_A2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1167514, upper bound: 3897.1170490
NS_A2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1168553, upper bound: 3897.1170947
NS_A2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1168553, upper bound: 3897.1171014
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1170435, upper bound: 3897.1167514
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1170435, upper bound: 3897.1167513
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1170947, upper bound: 3897.1168552
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1170947, upper bound: 3897.1168553
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1166729, upper bound: 3897.1166729
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1166729, upper bound: 3897.1167128
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1167130, upper bound: 3897.1167679
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.98
Output dim: 0, lower bound: -3897.1167130, upper bound: 3897.1168233

## BFS NS instance: NS_A1_B1_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -1417.8165283, 2502.5986328, -1417.8165283, 2502.5986328, -3920.4150391, 3920.4150391
1: -250.3135223, 316.1636353, -250.3135223, 316.1636353, -566.4770508, 566.4770508
2: -190.3193970, 427.1272278, -190.3193970, 427.1272278, -617.4466553, 617.4466553
3: -199.2041016, 559.2181396, -199.2041016, 559.2181396, -758.4222412, 758.4222412
4: -163.8197632, 536.0412598, -163.8197632, 536.0412598, -699.8609619, 699.8609619

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_B1_A1_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1172190, upper bound: 3897.1171951
time: 0.71 seconds

## Relational analysis of NS_A1_B1_B1_A1_A1_B1_B2

### Relational analysis result of NS_A1_B1_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171950, upper bound: 3897.1171927
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -1417.8165283, 2502.5986328, -1428.6033936, 2519.1269531, -3936.9433594, 3931.2021484
1: -250.3135223, 316.1636353, -251.7466888, 318.2354431, -568.5488892, 567.9102783
2: -190.3193970, 427.1272278, -191.4405518, 429.7970581, -620.1163940, 618.5677490
3: -199.2041016, 559.2181396, -200.3177490, 563.2192383, -762.4233398, 759.5358887
4: -163.8197632, 536.0412598, -164.7749176, 539.3828125, -703.2025757, 700.8161621

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A1

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171953, upper bound: 3897.1172042
time: 0.72 seconds

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A2

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171950, upper bound: 3897.1171928
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -1428.6033936, 2519.1269531, -1417.8165283, 2502.5986328, -3931.2021484, 3936.9433594
1: -251.7466888, 318.2354431, -250.3135223, 316.1636353, -567.9102783, 568.5488892
2: -191.4405518, 429.7970581, -190.3193970, 427.1272278, -618.5677490, 620.1163940
3: -200.3177490, 563.2192383, -199.2041016, 559.2181396, -759.5358887, 762.4233398
4: -164.7749176, 539.3828125, -163.8197632, 536.0412598, -700.8161621, 703.2025757

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_B1_A1_A2_B1_B1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1172042, upper bound: 3897.1171996
time: 0.69 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B1_B2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171928, upper bound: 3897.1171981
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -1428.6033936, 2519.1269531, -1428.6033936, 2519.1269531, -3947.7304688, 3947.7304688
1: -251.7466888, 318.2354431, -251.7466888, 318.2354431, -569.9820557, 569.9821167
2: -191.4405518, 429.7970581, -191.4405518, 429.7970581, -621.2374878, 621.2374878
3: -200.3177490, 563.2192383, -200.3177490, 563.2192383, -763.5369873, 763.5369873
4: -164.7749176, 539.3828125, -164.7749176, 539.3828125, -704.1577148, 704.1577148

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1172042, upper bound: 3897.1171996
time: 0.78 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171928, upper bound: 3897.1171981
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -1441.4382324, 2534.6342773, -1228.2359619, 2177.3125000, -3618.7507324, 3762.8693848
1: -252.8173828, 320.5074158, -217.7520905, 274.7200012, -527.5373535, 538.2595215
2: -192.5926971, 434.2507019, -165.4601440, 368.4635620, -561.0560913, 599.7108154
3: -202.3173981, 568.0910645, -173.2222290, 485.3515320, -687.6689453, 741.3131104
4: -165.8771973, 545.0367432, -142.3264771, 462.4692993, -628.3464966, 687.3632202

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_B1_A2_A1_B1_A1

### Relational analysis result of NS_A1_B1_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171709, upper bound: 3897.1170020
time: 0.65 seconds

## Relational analysis of NS_A1_B1_B1_A2_A1_B1_A2

### Relational analysis result of NS_A1_B1_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171709, upper bound: 3897.1170019
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -1455.4689941, 2558.6281738, -1396.2453613, 2457.4299316, -3912.8979492, 3954.8735352
1: -255.1707153, 323.6001587, -245.1777802, 310.5320435, -565.7027588, 568.7778931
2: -194.4391022, 438.2991638, -186.7776642, 419.4938049, -613.9328613, 625.0768433
3: -204.2024841, 573.5820312, -195.8167267, 549.6403809, -753.8428955, 769.3986206
4: -167.3996429, 550.1965332, -160.7947845, 526.4590454, -693.8587036, 710.9913330

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_B1_A2_A1_B2_A1

### Relational analysis result of NS_A1_B1_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171709, upper bound: 3897.1170020
time: 0.64 seconds

## Relational analysis of NS_A1_B1_B1_A2_A1_B2_A2

### Relational analysis result of NS_A1_B1_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171709, upper bound: 3897.1170020
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -1476.4704590, 2595.3466797, -1417.8165283, 2502.5986328, -3979.0686035, 4013.1628418
1: -258.8100281, 328.3325806, -250.3135223, 316.1636353, -574.9736328, 578.6460571
2: -197.2351990, 444.0037842, -190.3193970, 427.1272278, -624.3624268, 634.3231812
3: -207.0301361, 581.8175659, -199.2041016, 559.2181396, -766.2482910, 781.0216675
4: -169.7957153, 557.4502563, -163.8197632, 536.0412598, -705.8368530, 721.2699585

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_B1_A2_A2_B1_B1

### Relational analysis result of NS_A1_B1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171728, upper bound: 3897.1170501
time: 0.76 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2_B1_B2

### Relational analysis result of NS_A1_B1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171696, upper bound: 3897.1170518
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -1476.4704590, 2595.3466797, -1428.6033936, 2519.1269531, -3995.5966797, 4023.9501953
1: -258.8100281, 328.3325806, -251.7466888, 318.2354431, -577.0454712, 580.0792847
2: -197.2351990, 444.0037842, -191.4405518, 429.7970581, -627.0321655, 635.4442749
3: -207.0301361, 581.8175659, -200.3177490, 563.2192383, -770.2493896, 782.1353149
4: -169.7957153, 557.4502563, -164.7749176, 539.3828125, -709.1785278, 722.2251587

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_B1_A2_A2_B2_B1

### Relational analysis result of NS_A1_B1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171728, upper bound: 3897.1170513
time: 0.75 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2_B2_B2

### Relational analysis result of NS_A1_B1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171697, upper bound: 3897.1170518
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1228.2359619, 2177.3125000, -1441.4382324, 2534.6342773, -3762.8693848, 3618.7507324
1: -217.7520905, 274.7200012, -252.8173828, 320.5074158, -538.2595215, 527.5373535
2: -165.4601440, 368.4635620, -192.5926971, 434.2507019, -599.7108154, 561.0560913
3: -173.2222290, 485.3515320, -202.3173981, 568.0910645, -741.3131104, 687.6689453
4: -142.3264771, 462.4692993, -165.8771973, 545.0367432, -687.3632202, 628.3464355

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170020, upper bound: 3897.1171709
time: 0.68 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170020, upper bound: 3897.1171707
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1396.2453613, 2457.4299316, -1455.4689941, 2558.6281738, -3954.8735352, 3912.8979492
1: -245.1777802, 310.5320435, -255.1707153, 323.6001587, -568.7778931, 565.7027588
2: -186.7776642, 419.4938049, -194.4391022, 438.2991638, -625.0767822, 613.9328613
3: -195.8167267, 549.6403809, -204.2024841, 573.5820312, -769.3986206, 753.8428955
4: -160.7947845, 526.4590454, -167.3996429, 550.1965332, -710.9913330, 693.8587036

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170020, upper bound: 3897.1171709
time: 0.66 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170020, upper bound: 3897.1171708
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1417.8165283, 2502.5986328, -1476.4704590, 2595.3466797, -4013.1630859, 3979.0686035
1: -250.3135223, 316.1636353, -258.8100281, 328.3325806, -578.6460571, 574.9736328
2: -190.3193970, 427.1272278, -197.2351990, 444.0037842, -634.3231812, 624.3624268
3: -199.2041016, 559.2181396, -207.0301361, 581.8175659, -781.0216675, 766.2482910
4: -163.8197632, 536.0412598, -169.7957153, 557.4502563, -721.2700195, 705.8368530

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_B2_A1_B2_A1_A1

### Relational analysis result of NS_A1_B1_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170034, upper bound: 3897.1171727
time: 0.68 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2_A1_A2

### Relational analysis result of NS_A1_B1_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170020, upper bound: 3897.1171695
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1428.6033936, 2519.1269531, -1476.4704590, 2595.3466797, -4023.9501953, 3995.5966797
1: -251.7466888, 318.2354431, -258.8100281, 328.3325806, -580.0792847, 577.0454712
2: -191.4405518, 429.7970581, -197.2351990, 444.0037842, -635.4442749, 627.0321655
3: -200.3177490, 563.2192383, -207.0301361, 581.8175659, -782.1353149, 770.2493286
4: -164.7749176, 539.3828125, -169.7957153, 557.4502563, -722.2251587, 709.1785278

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_B2_A1_B2_A2_A1

### Relational analysis result of NS_A1_B1_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170034, upper bound: 3897.1172004
time: 0.71 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2_A2_A2

### Relational analysis result of NS_A1_B1_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170020, upper bound: 3897.1171783
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -1476.8770752, 2599.3308105, -1476.8770752, 2599.3308105, -4076.2080078, 4076.2080078
1: -259.4418030, 328.6999207, -259.4418030, 328.6999207, -588.1417236, 588.1417236
2: -197.5400696, 444.9430542, -197.5400696, 444.9430542, -642.4831543, 642.4831543
3: -207.2708130, 582.3969116, -207.2708130, 582.3969116, -789.6677246, 789.6677246
4: -170.0515594, 558.5493164, -170.0515594, 558.5493164, -728.6008911, 728.6008911

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_B2_A2_A1_B1_A1

### Relational analysis result of NS_A1_B1_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1169896, upper bound: 3897.1170805
time: 0.66 seconds

## Relational analysis of NS_A1_B1_B2_A2_A1_B1_A2

### Relational analysis result of NS_A1_B1_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1169780, upper bound: 3897.1169780
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -1476.8770752, 2599.3308105, -1476.4704590, 2595.3466797, -4072.2233887, 4075.8010254
1: -259.4418030, 328.6999207, -258.8100281, 328.3325806, -587.7744141, 587.5099487
2: -197.5400696, 444.9430542, -197.2351990, 444.0037842, -641.5438232, 642.1782227
3: -207.2708130, 582.3969116, -207.0301361, 581.8175659, -789.0883789, 789.4270630
4: -170.0515594, 558.5493164, -169.7957153, 557.4502563, -727.5017700, 728.3449707

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_B2_A2_A1_B2_B1

### Relational analysis result of NS_A1_B1_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170855, upper bound: 3897.1169986
time: 0.76 seconds

## Relational analysis of NS_A1_B1_B2_A2_A1_B2_B2

### Relational analysis result of NS_A1_B1_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1169780, upper bound: 3897.1169826
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -1476.4704590, 2595.3466797, -1476.8770752, 2599.3308105, -4075.8010254, 4072.2233887
1: -258.8100281, 328.3325806, -259.4418030, 328.6999207, -587.5099487, 587.7743530
2: -197.2351990, 444.0037842, -197.5400696, 444.9430542, -642.1782227, 641.5438232
3: -207.0301361, 581.8175659, -207.2708130, 582.3969116, -789.4270630, 789.0883789
4: -169.7957153, 557.4502563, -170.0515594, 558.5493164, -728.3449707, 727.5018311

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_B2_A2_A2_B1_A1

### Relational analysis result of NS_A1_B1_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1169986, upper bound: 3897.1171395
time: 0.74 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2_B1_A2

### Relational analysis result of NS_A1_B1_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1169824, upper bound: 3897.1170202
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -1476.4704590, 2595.3466797, -1476.4704590, 2595.3466797, -4071.8164062, 4071.8164062
1: -258.8100281, 328.3325806, -258.8100281, 328.3325806, -587.1425781, 587.1425781
2: -197.2351990, 444.0037842, -197.2351990, 444.0037842, -641.2390137, 641.2390137
3: -207.0301361, 581.8175659, -207.0301361, 581.8175659, -788.8477173, 788.8477173
4: -169.7957153, 557.4502563, -169.7957153, 557.4502563, -727.2459106, 727.2459106

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_A1

### Relational analysis result of NS_A1_B1_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1169986, upper bound: 3897.1171659
time: 0.71 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1169824, upper bound: 3897.1170350
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -1431.8116455, 2526.5187988, -1560.8592529, 2758.1225586, -4189.9340820, 4087.3779297
1: -252.5830994, 319.2783508, -274.9635620, 349.1387329, -601.7216797, 594.2419434
2: -192.0289001, 430.9742126, -209.4033051, 468.0413818, -660.0703125, 640.3775024
3: -200.9347076, 564.7389526, -219.0403290, 616.7258911, -817.6605835, 783.7792969
4: -165.2748108, 540.8487549, -179.9197693, 587.6668701, -752.9415283, 720.7684937

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_B1_A1_B1_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168865, upper bound: 3897.1168612
time: 0.70 seconds

## Relational analysis of NS_A1_B2_B1_A1_B1_B1_A2

### Relational analysis result of NS_A1_B2_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168865, upper bound: 3897.1168613
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -1382.0690918, 2438.8747559, -1679.6984863, 2959.8737793, -4341.9418945, 4118.5732422
1: -243.5602417, 307.9483032, -294.6857300, 374.6565247, -618.2167358, 602.6340332
2: -185.3246002, 415.3713989, -224.8072205, 503.4781799, -688.8027954, 640.1785889
3: -194.0104980, 544.5928345, -235.8617554, 662.3085938, -856.3190308, 780.4545898
4: -159.5112762, 521.2068481, -193.3466492, 631.5406494, -791.0519409, 714.5534668

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_B1_A1_B1_B2_A1

### Relational analysis result of NS_A1_B2_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168865, upper bound: 3897.1168612
time: 0.81 seconds

## Relational analysis of NS_A1_B2_B1_A1_B1_B2_A2

### Relational analysis result of NS_A1_B2_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168865, upper bound: 3897.1168613
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -1342.9747314, 2365.4709473, -2598.8002930, 4559.1938477, -5896.7675781, 4964.2714844
1: -236.3076782, 298.9551086, -451.4018250, 577.8993530, -814.1426392, 750.3566895
2: -179.8934021, 402.7038574, -347.2883606, 779.9246826, -958.1946411, 749.9921875
3: -188.6383362, 528.6361694, -362.9587097, 1023.9326782, -1209.6560059, 891.5948486
4: -154.8398895, 505.5078735, -298.8435669, 978.5938721, -1130.3817139, 804.3514404

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_B1_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1167292, upper bound: 3897.1169958
time: 0.68 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B1_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1166666, upper bound: 3897.1169424
time: 0.73 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1166722, upper bound: 3897.1169737
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -1329.7253418, 2343.9741211, -2543.0058594, 4461.2294922, -5787.7939453, 4886.9790039
1: -234.1968994, 296.0747681, -441.8514709, 565.1511230, -799.3480225, 737.9262085
2: -178.2221985, 398.1972046, -339.8407288, 762.4144287, -939.8407593, 738.0379639
3: -186.8201752, 523.3226318, -355.3584290, 1001.4480591, -1186.4359131, 878.6809082
4: -153.3820496, 499.9361877, -292.4694214, 956.6247559, -1108.0809326, 792.4056396

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_B1_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170297, upper bound: 3897.1170386
time: 0.75 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170417, upper bound: 3897.1170594
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -1464.1042480, 2576.3720703, -1529.3286133, 2702.3383789, -4166.4423828, 4105.7006836
1: -257.1084595, 325.8310547, -269.4979553, 341.8139343, -598.9223022, 595.3289795
2: -195.7712708, 440.8126831, -205.2101898, 459.0283813, -654.7996826, 646.0228882
3: -205.3856201, 577.3152466, -214.6475983, 604.1166382, -809.5022583, 791.9628296
4: -168.5271912, 553.3787231, -176.3538208, 576.2686768, -744.7958374, 729.7324829

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_B1_A2_B1_B1_B1

### Relational analysis result of NS_A1_B2_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1169299, upper bound: 3897.1166870
time: 0.61 seconds

## Relational analysis of NS_A1_B2_B1_A2_B1_B1_B2

### Relational analysis result of NS_A1_B2_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1170343, upper bound: 3897.1168969
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -1386.6691895, 2437.1855469, -2607.1340332, 4575.3657227, -5957.9350586, 5044.3193359
1: -243.0972290, 308.2707825, -453.2289734, 579.9715576, -823.0687256, 761.4996948
2: -185.2105713, 416.6575928, -348.6174316, 783.1285400, -967.1853027, 765.2749634
3: -194.5843048, 546.2713623, -364.0946045, 1027.4166260, -1219.5083008, 910.3659058
4: -159.3973694, 523.2362061, -299.9079895, 982.5173950, -1139.3111572, 823.1441040

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_B1_A2_B1_B2_A1

### Relational analysis result of NS_A1_B2_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171069, upper bound: 3897.1170472
time: 0.62 seconds

## Relational analysis of NS_A1_B2_B1_A2_B1_B2_A2

### Relational analysis result of NS_A1_B2_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1171069, upper bound: 3897.1170472
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1476.8770752, 2599.3308105, -1617.9702148, 2852.7858887, -4329.6630859, 4217.3007812
1: -259.4418030, 328.6999207, -284.1601868, 361.2568970, -620.6987305, 612.8600464
2: -197.5400696, 444.9430542, -216.7185364, 484.5160217, -682.0560913, 661.6614990
3: -207.2708130, 582.3969116, -226.9315186, 638.5805664, -845.8513794, 809.3284302
4: -170.0515594, 558.5493164, -186.2865601, 608.2865601, -778.3379517, 744.8358765

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1164717, upper bound: 3897.1156505
time: 0.74 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1167172, upper bound: 3897.1156861
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1476.4704590, 2595.3466797, -1617.9702148, 2852.7858887, -4329.2563477, 4213.3164062
1: -258.8100281, 328.3325806, -284.1601868, 361.2568970, -620.0668945, 612.4926758
2: -197.2351990, 444.0037842, -216.7185364, 484.5160217, -681.7512207, 660.7222900
3: -207.0301361, 581.8175659, -226.9315186, 638.5805664, -845.6107178, 808.7490845
4: -169.7957153, 557.4502563, -186.2865601, 608.2865601, -778.0821533, 743.7368164

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1164717, upper bound: 3897.1161116
time: 0.74 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1167172, upper bound: 3897.1161328
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -1413.7648926, 2493.2202148, -1550.0106201, 2728.6325684, -4142.3969727, 4043.2309570
1: -249.2973328, 315.0601196, -271.6847229, 345.5222778, -594.8195801, 586.7448730
2: -189.5510254, 425.5280151, -207.2277832, 464.2865906, -653.8376465, 632.7557983
3: -198.4967041, 557.4090576, -217.7178040, 611.1742554, -809.6707764, 775.1268311
4: -163.1852570, 534.0260010, -178.2461853, 583.1766357, -746.3618774, 712.2722168

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B2_A1_B1_B1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1109310, upper bound: 3897.1131333
time: 0.73 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_B2_A1_B1_B1_A1

### Relational analysis result of NS_A1_B2_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1125532, upper bound: 3897.1143217
time: 0.63 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B2_A1_B1_B1_A1

### Relational analysis result of NS_A1_B2_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1128104, upper bound: 3897.1144438
time: 0.65 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_B2_A1_B1_B1_A1

### Relational analysis result of NS_A1_B2_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1129885, upper bound: 3897.1145315
time: 0.77 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_B1_A2

### Relational analysis result of NS_A1_B2_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1129885, upper bound: 3897.1145315
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -1396.2578125, 2463.7260742, -1530.9134521, 2691.6928711, -4087.9494629, 3994.6396484
1: -246.3226624, 311.1182556, -268.2526855, 340.9388123, -587.2614746, 579.3709717
2: -187.2419586, 419.7803650, -204.6337280, 457.6206055, -644.8625488, 624.4140625
3: -196.0427551, 550.2819824, -215.2246246, 603.0212402, -799.0639648, 765.5065918
4: -161.1918945, 526.9128418, -176.0340424, 574.7609863, -735.9528198, 702.9468384

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B2_A1_B1_B2_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1159227, upper bound: 3897.1166703
time: 0.70 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_B2_B2

### Relational analysis result of NS_A1_B2_B2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.0935107, upper bound: 3897.1133039
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -1339.0776367, 2360.9387207, -2673.6008301, 4684.3168945, -6020.7011719, 5034.5390625
1: -235.8031616, 298.2247314, -463.5070801, 593.8083496, -829.6115112, 761.7317505
2: -179.5122223, 401.5906677, -356.8432617, 802.4544067, -980.8639526, 758.4339600
3: -188.0821533, 527.2203369, -373.2714539, 1052.6611328, -1238.5537109, 900.4917603
4: -154.4908600, 504.0829468, -307.0988464, 1006.8601074, -1159.0222168, 811.1817627

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_B2_A1_B2_B1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1158453, upper bound: 3897.1165838
time: 0.67 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_B1_B2

### Relational analysis result of NS_A1_B2_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1165627, upper bound: 3897.1169276
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -1350.7188721, 2381.0168457, -2697.0847168, 4724.3100586, -6073.9125977, 5078.1015625
1: -237.8354187, 300.8186646, -467.1334229, 598.8817749, -836.7171631, 767.9520874
2: -180.9874420, 405.2300415, -359.7681274, 808.7622681, -988.8341675, 764.9980469
3: -189.5330200, 531.9595337, -376.4081421, 1061.7937012, -1249.4808350, 908.3674927
4: -155.7412872, 508.6641541, -309.6517944, 1014.8596191, -1168.4346924, 818.3159180

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_B2_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168405, upper bound: 3897.1171138
time: 0.71 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3897.1168532, upper bound: 3897.1171258
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -1464.6647949, 2575.1318359, -1550.0106201, 2728.6325684, -4193.2968750, 4125.1425781
1: -256.9198914, 325.7557373, -271.6847229, 345.5222778, -602.4421387, 597.4404297
2: -195.7091675, 440.5400085, -207.2277832, 464.2865906, -659.9957275, 647.7678223
3: -205.5114441, 577.1492310, -217.7178040, 611.1742554, -816.6856689, 794.8670044
4: -168.5133057, 553.0742188, -178.2461853, 583.1766357, -751.6898804, 731.3204346

Time for backsubstitution: 2.42 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.28 + 418.04 = 422.32 seconds
