## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 6)
Time budget: 420 seconds
Split limit: 100
Threshold: 3295.06222004829


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6975.3520508, 12629.0117188, -6975.3520508, 12629.0117188, -19604.3632812, 19604.3632812)
1: (-1367.2161865, 1793.9888916, -1367.2161865, 1793.9888916, -3161.2050781, 3161.2050781)
2: (-1009.3267822, 2079.8847656, -1009.3267822, 2079.8847656, -3089.2114258, 3089.2114258)
3: (-1079.8317871, 2775.1779785, -1079.8317871, 2775.1779785, -3855.0090332, 3855.0090332)
4: (-860.8695068, 2594.6042480, -860.8695068, 2594.6042480, -3455.4736328, 3455.4736328)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.35 + 2.19 = 4.54 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -3295.0951710, upper bound: 3295.0951709

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0936399, upper bound: 3295.0929875
time: 0.88 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0936996, upper bound: 3295.0936995
time: 0.82 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.91 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 1.91
Output dim: 3, lower bound: -3295.0936399, upper bound: 3295.0929875
NS_B2, status: Status.UNKNOWN, split count: 1, time: 1.91
Output dim: 3, lower bound: -3295.0936996, upper bound: 3295.0936995

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -6805.2792969, 12325.2939453, -6712.3061523, 12160.8310547, -18966.1093750, 19037.5996094
1: -1333.6514893, 1750.5275879, -1315.4766846, 1726.8565674, -3060.5080566, 3066.0041504
2: -984.8374634, 2029.6968994, -971.5372925, 2002.2072754, -2987.0446777, 3001.2341309
3: -1053.1970215, 2708.7565918, -1038.6553955, 2672.6721191, -3725.8691406, 3747.4116211
4: -839.8204956, 2532.4892578, -828.3814697, 2498.4587402, -3338.2788086, 3360.8706055

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0929875, upper bound: 3295.0929875
time: 0.83 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0929875, upper bound: 3295.0929875
time: 0.89 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -6870.3359375, 12434.9541016, -7417.5991211, 13374.9853516, -20245.3203125, 19852.5527344
1: -1346.1413574, 1766.5756836, -1441.8498535, 1896.5756836, -3242.7170410, 3208.4252930
2: -993.8072510, 2047.4963379, -1068.5217285, 2199.3256836, -3193.1328125, 3116.0178223
3: -1063.4912109, 2732.8493652, -1144.7545166, 2944.7966309, -4008.2878418, 3877.6037598
4: -847.5876465, 2554.4721680, -911.7056885, 2743.9956055, -3591.5825195, 3466.1774902

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0929875, upper bound: 3295.0936397
time: 0.84 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0929875, upper bound: 3295.0936397
time: 0.93 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.12 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 4.12
Output dim: 3, lower bound: -3295.0929875, upper bound: 3295.0929875
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 4.12
Output dim: 3, lower bound: -3295.0929875, upper bound: 3295.0929875
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 4.12
Output dim: 3, lower bound: -3295.0929875, upper bound: 3295.0936397
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 4.12
Output dim: 3, lower bound: -3295.0929875, upper bound: 3295.0936397

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -6712.3061523, 12160.8310547, -6712.3061523, 12160.8310547, -18873.1367188, 18873.1367188
1: -1315.4766846, 1726.8565674, -1315.4766846, 1726.8565674, -3042.3330078, 3042.3330078
2: -971.5372925, 2002.2072754, -971.5372925, 2002.2072754, -2973.7446289, 2973.7446289
3: -1038.6553955, 2672.6721191, -1038.6553955, 2672.6721191, -3711.3276367, 3711.3276367
4: -828.3814697, 2498.4587402, -828.3814697, 2498.4587402, -3326.8400879, 3326.8400879

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_B1

### Relational analysis result of NS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0929317, upper bound: 3295.0929532
time: 0.80 seconds

## Relational analysis of NS_B1_A1_B2

### Relational analysis result of NS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0929307, upper bound: 3295.0929282
time: 0.77 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -7417.5991211, 13374.9853516, -6712.3061523, 12160.8310547, -19578.4296875, 20087.2910156
1: -1441.8498535, 1896.5756836, -1315.4766846, 1726.8565674, -3168.7060547, 3212.0522461
2: -1068.5217285, 2199.3256836, -971.5372925, 2002.2072754, -3070.7290039, 3170.8630371
3: -1144.7545166, 2944.7966309, -1038.6553955, 2672.6721191, -3817.4267578, 3983.4521484
4: -911.7056885, 2743.9956055, -828.3814697, 2498.4587402, -3410.1640625, 3572.3769531

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0927675, upper bound: 3295.0929160
time: 0.81 seconds

## Relational analysis of NS_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0927644, upper bound: 3295.0927644
time: 0.83 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -6712.3061523, 12160.8310547, -7417.5991211, 13374.9853516, -20087.2910156, 19578.4296875
1: -1315.4766846, 1726.8565674, -1441.8498535, 1896.5756836, -3212.0522461, 3168.7060547
2: -971.5372925, 2002.2072754, -1068.5217285, 2199.3256836, -3170.8630371, 3070.7290039
3: -1038.6553955, 2672.6721191, -1144.7545166, 2944.7966309, -3983.4521484, 3817.4267578
4: -828.3814697, 2498.4587402, -911.7056885, 2743.9956055, -3572.3769531, 3410.1640625

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_A1_A1

### Relational analysis result of NS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0929159, upper bound: 3295.0933574
time: 0.83 seconds

## Relational analysis of NS_B2_A1_A2

### Relational analysis result of NS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0927644, upper bound: 3295.0933538
time: 1.19 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -7417.5991211, 13374.9853516, -7417.5991211, 13374.9853516, -20792.5839844, 20792.5839844
1: -1441.8498535, 1896.5756836, -1441.8498535, 1896.5756836, -3338.4250488, 3338.4250488
2: -1068.5217285, 2199.3256836, -1068.5217285, 2199.3256836, -3267.8474121, 3267.8474121
3: -1144.7545166, 2944.7966309, -1144.7545166, 2944.7966309, -4089.5512695, 4089.5512695
4: -911.7056885, 2743.9956055, -911.7056885, 2743.9956055, -3655.7009277, 3655.7009277

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_B1

### Relational analysis result of NS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0929323, upper bound: 3295.0933208
time: 0.76 seconds

## Relational analysis of NS_B2_A2_B2

### Relational analysis result of NS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0929282, upper bound: 3295.0934993
time: 0.80 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.41 seconds
NS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.41
Output dim: 3, lower bound: -3295.0929317, upper bound: 3295.0929532
NS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.41
Output dim: 3, lower bound: -3295.0929307, upper bound: 3295.0929282
NS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.41
Output dim: 3, lower bound: -3295.0927675, upper bound: 3295.0929160
NS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 4.41
Output dim: 3, lower bound: -3295.0927644, upper bound: 3295.0927644
NS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 4.41
Output dim: 3, lower bound: -3295.0929159, upper bound: 3295.0933574
NS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 4.41
Output dim: 3, lower bound: -3295.0927644, upper bound: 3295.0933538
NS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.41
Output dim: 3, lower bound: -3295.0929323, upper bound: 3295.0933208
NS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 4.41
Output dim: 3, lower bound: -3295.0929282, upper bound: 3295.0934993

## BFS NS instance: NS_B1_A1_B1

### Backsubstitution after applying NS history:
0: -6622.4404297, 11993.2734375, -6580.8979492, 11916.5761719, -18539.0156250, 18574.1718750
1: -1297.5631104, 1703.5190430, -1289.3294678, 1692.7764893, -2990.3395996, 2992.8479004
2: -958.2832031, 1974.9407959, -952.1642456, 1962.3341064, -2920.6171875, 2927.1049805
3: -1024.6959229, 2635.7988281, -1018.2354126, 2618.9416504, -3643.6376953, 3654.0341797
4: -817.0397339, 2464.6904297, -811.8054810, 2449.0734863, -3266.1132812, 3276.4953613

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_B1_A1

### Relational analysis result of NS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0929308, upper bound: 3295.0929308
time: 0.88 seconds

## Relational analysis of NS_B1_A1_B1_A2

### Relational analysis result of NS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0929308, upper bound: 3295.0929308
time: 0.80 seconds

## BFS NS instance: NS_B1_A1_B2

### Backsubstitution after applying NS history:
0: -6633.4233398, 12017.8183594, -6738.9633789, 12202.0117188, -18835.4316406, 18756.7812500
1: -1300.0958252, 1706.6977539, -1320.1705322, 1732.3896484, -3032.4853516, 3026.8681641
2: -960.1320801, 1978.5385742, -974.7658081, 2009.0344238, -2969.1665039, 2953.3044434
3: -1026.4858398, 2640.8723145, -1042.8737793, 2681.6857910, -3708.1713867, 3683.7460938
4: -818.6002808, 2469.2087402, -831.2281494, 2507.3364258, -3325.9365234, 3300.4367676

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_B2_A1

### Relational analysis result of NS_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0929308, upper bound: 3295.0929308
time: 0.80 seconds

## Relational analysis of NS_B1_A1_B2_A2

### Relational analysis result of NS_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0929308, upper bound: 3295.0929306
time: 0.86 seconds

## BFS NS instance: NS_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7380.4414062, 13306.7089844, -6612.0605469, 11977.6464844, -19358.0820312, 19918.7695312
1: -1434.3153076, 1886.8489990, -1295.3778076, 1700.6385498, -3134.9536133, 3182.2268066
2: -1063.0360107, 2188.1284180, -956.7649536, 1972.0057373, -3035.0417480, 3144.8930664
3: -1138.8378906, 2929.9018555, -1022.7341919, 2632.6567383, -3771.4946289, 3952.6359863
4: -907.0045166, 2730.0686035, -815.7456665, 2460.8723145, -3367.8769531, 3545.8142090

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B1_A2_B1_B1

### Relational analysis result of NS_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0932504, upper bound: 3295.0928603
time: 0.83 seconds

## Relational analysis of NS_B1_A2_B1_B2

### Relational analysis result of NS_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0932540, upper bound: 3295.0928177
time: 0.85 seconds

## BFS NS instance: NS_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7325.3457031, 13204.9316406, -7938.3154297, 14408.0986328, -21733.4433594, 21143.2460938
1: -1423.4370117, 1872.5198975, -1557.6616211, 2037.6036377, -3461.0405273, 3430.1816406
2: -1055.2239990, 2172.8593750, -1149.2247314, 2369.4418945, -3424.6660156, 3322.0839844
3: -1130.7808838, 2907.7934570, -1229.1257324, 3166.1289062, -4296.9096680, 4136.9184570
4: -900.4540405, 2711.0476074, -981.2775879, 2953.6350098, -3854.0886230, 3692.3247070

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B2_A1

### Relational analysis result of NS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0927434, upper bound: 3295.0920762
time: 0.84 seconds

## Relational analysis of NS_B1_A2_B2_A2

### Relational analysis result of NS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0898563, upper bound: 3295.0908872
time: 0.83 seconds

## BFS NS instance: NS_B2_A1_A1

### Backsubstitution after applying NS history:
0: -6612.0605469, 11977.6464844, -7380.4414062, 13306.7089844, -19918.7695312, 19358.0820312
1: -1295.3778076, 1700.6385498, -1434.3153076, 1886.8489990, -3182.2268066, 3134.9536133
2: -956.7649536, 1972.0057373, -1063.0360107, 2188.1284180, -3144.8930664, 3035.0417480
3: -1022.7341919, 2632.6567383, -1138.8378906, 2929.9018555, -3952.6359863, 3771.4946289
4: -815.7456665, 2460.8723145, -907.0045166, 2730.0686035, -3545.8142090, 3367.8769531

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A1_A1_A1

### Relational analysis result of NS_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0928603, upper bound: 3295.0932504
time: 0.78 seconds

## Relational analysis of NS_B2_A1_A1_A2

### Relational analysis result of NS_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0928177, upper bound: 3295.0932540
time: 0.84 seconds

## BFS NS instance: NS_B2_A1_A2

### Backsubstitution after applying NS history:
0: -7938.3154297, 14408.0986328, -7325.3457031, 13204.9316406, -21143.2460938, 21733.4453125
1: -1557.6616211, 2037.6036377, -1423.4370117, 1872.5198975, -3430.1816406, 3461.0405273
2: -1149.2247314, 2369.4418945, -1055.2239990, 2172.8593750, -3322.0839844, 3424.6660156
3: -1229.1257324, 3166.1289062, -1130.7808838, 2907.7934570, -4136.9189453, 4296.9096680
4: -981.2775879, 2953.6350098, -900.4540405, 2711.0476074, -3692.3247070, 3854.0886230

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B2_A1_A2_B1

### Relational analysis result of NS_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0920762, upper bound: 3295.0927436
time: 0.82 seconds

## Relational analysis of NS_B2_A1_A2_B2

### Relational analysis result of NS_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0908873, upper bound: 3295.0898564
time: 0.82 seconds

## BFS NS instance: NS_B2_A2_B1

### Backsubstitution after applying NS history:
0: -7328.0039062, 13211.1542969, -7283.7900391, 13129.8017578, -20457.8046875, 20494.9433594
1: -1424.5435791, 1873.5737305, -1415.9500732, 1862.2276611, -3286.7712402, 3289.5234375
2: -1055.5045166, 2172.7104492, -1049.0753174, 2159.5200195, -3215.0244141, 3221.7856445
3: -1131.0288086, 2908.7990723, -1124.2354736, 2890.9475098, -4021.9763184, 4033.0341797
4: -900.5809937, 2711.0214844, -895.0813599, 2694.6889648, -3595.2700195, 3606.1025391

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_B1_A1

### Relational analysis result of NS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0933233, upper bound: 3295.0933209
time: 0.83 seconds

## Relational analysis of NS_B2_A2_B1_A2

### Relational analysis result of NS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0933233, upper bound: 3295.0933209
time: 0.84 seconds

## BFS NS instance: NS_B2_A2_B2

### Backsubstitution after applying NS history:
0: -7328.0815430, 13210.2783203, -7364.7934570, 13272.5234375, -20600.6035156, 20575.0722656
1: -1424.3073730, 1873.6419678, -1431.2687988, 1882.0081787, -3306.3154297, 3304.9106445
2: -1055.5369873, 2172.5622559, -1060.3488770, 2182.7976074, -3238.3344727, 3232.9108887
3: -1130.9757080, 2908.6999512, -1136.6070557, 2922.7785645, -4053.7541504, 4045.3071289
4: -900.5895996, 2710.8840332, -904.7531128, 2723.8620605, -3624.4516602, 3615.6369629

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_B2_A1

### Relational analysis result of NS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0933238, upper bound: 3295.0934994
time: 0.84 seconds

## Relational analysis of NS_B2_A2_B2_A2

### Relational analysis result of NS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0933237, upper bound: 3295.0934994
time: 0.85 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.09 seconds
NS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -3295.0929308, upper bound: 3295.0929308
NS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -3295.0929308, upper bound: 3295.0929308
NS_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -3295.0929308, upper bound: 3295.0929308
NS_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -3295.0929308, upper bound: 3295.0929306
NS_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -3295.0932504, upper bound: 3295.0928603
NS_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -3295.0932540, upper bound: 3295.0928177
NS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -3295.0927434, upper bound: 3295.0920762
NS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -3295.0898563, upper bound: 3295.0908872
NS_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -3295.0928603, upper bound: 3295.0932504
NS_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -3295.0928177, upper bound: 3295.0932540
NS_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -3295.0920762, upper bound: 3295.0927436
NS_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -3295.0908873, upper bound: 3295.0898564
NS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -3295.0933233, upper bound: 3295.0933209
NS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -3295.0933233, upper bound: 3295.0933209
NS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -3295.0933238, upper bound: 3295.0934994
NS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -3295.0933237, upper bound: 3295.0934994

## BFS NS instance: NS_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -6580.8979492, 11916.5761719, -6580.8979492, 11916.5761719, -18497.4746094, 18497.4746094
1: -1289.3294678, 1692.7764893, -1289.3294678, 1692.7764893, -2982.1059570, 2982.1059570
2: -952.1642456, 1962.3341064, -952.1642456, 1962.3341064, -2914.4982910, 2914.4980469
3: -1018.2354126, 2618.9416504, -1018.2354126, 2618.9416504, -3637.1770020, 3637.1770020
4: -811.8054810, 2449.0734863, -811.8054810, 2449.0734863, -3260.8789062, 3260.8789062

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_B1_A1_B1

### Relational analysis result of NS_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0816298, upper bound: 3295.0871327
time: 0.72 seconds

## Relational analysis of NS_B1_A1_B1_A1_B2

### Relational analysis result of NS_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0816256, upper bound: 3295.0816249
time: 0.89 seconds

## BFS NS instance: NS_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -6738.9633789, 12202.0117188, -6580.8979492, 11916.5761719, -18655.5390625, 18782.9062500
1: -1320.1705322, 1732.3896484, -1289.3294678, 1692.7764893, -3012.9470215, 3021.7192383
2: -974.7658081, 2009.0344238, -952.1642456, 1962.3341064, -2937.0998535, 2961.1987305
3: -1042.8737793, 2681.6857910, -1018.2354126, 2618.9416504, -3661.8154297, 3699.9206543
4: -831.2281494, 2507.3364258, -811.8054810, 2449.0734863, -3280.3017578, 3319.1416016

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_B1_A2_B1

### Relational analysis result of NS_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0911344, upper bound: 3295.0925121
time: 1.07 seconds

## Relational analysis of NS_B1_A1_B1_A2_B2

### Relational analysis result of NS_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0911366, upper bound: 3295.0913249
time: 0.84 seconds

## BFS NS instance: NS_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6580.8979492, 11916.5761719, -6738.9633789, 12202.0117188, -18782.9062500, 18655.5390625
1: -1289.3294678, 1692.7764893, -1320.1705322, 1732.3896484, -3021.7192383, 3012.9470215
2: -952.1642456, 1962.3341064, -974.7658081, 2009.0344238, -2961.1984863, 2937.0998535
3: -1018.2354126, 2618.9416504, -1042.8737793, 2681.6857910, -3699.9206543, 3661.8154297
4: -811.8054810, 2449.0734863, -831.2281494, 2507.3364258, -3319.1416016, 3280.3017578

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_B2_A1_A1

### Relational analysis result of NS_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0923728, upper bound: 3295.0911339
time: 0.73 seconds

## Relational analysis of NS_B1_A1_B2_A1_A2

### Relational analysis result of NS_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0911353, upper bound: 3295.0911345
time: 0.84 seconds

## BFS NS instance: NS_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -6738.9633789, 12202.0117188, -6738.9633789, 12202.0117188, -18940.9726562, 18940.9726562
1: -1320.1705322, 1732.3896484, -1320.1705322, 1732.3896484, -3052.5600586, 3052.5600586
2: -974.7658081, 2009.0344238, -974.7658081, 2009.0344238, -2983.8002930, 2983.8002930
3: -1042.8737793, 2681.6857910, -1042.8737793, 2681.6857910, -3724.5590820, 3724.5590820
4: -831.2281494, 2507.3364258, -831.2281494, 2507.3364258, -3338.5644531, 3338.5644531

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_B2_A2_A1

### Relational analysis result of NS_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0869548, upper bound: 3295.0857480
time: 0.93 seconds

## Relational analysis of NS_B1_A1_B2_A2_A2

### Relational analysis result of NS_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0816250, upper bound: 3295.0857498
time: 0.82 seconds

## BFS NS instance: NS_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -7290.8300781, 13142.9208984, -6483.1650391, 11737.7138672, -19028.5429688, 19626.0820312
1: -1417.0169678, 1863.8448486, -1269.7789307, 1667.1958008, -3084.2128906, 3133.6237793
2: -1050.0134277, 2161.5070801, -937.7656860, 1932.8701172, -2982.8833008, 3099.2724609
3: -1125.1123047, 2893.9077148, -1002.7440186, 2579.8486328, -3704.9609375, 3896.6511230
4: -895.8804321, 2697.0876465, -799.5012817, 2412.3791504, -3308.2595215, 3496.5883789

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A2_B1_B1_A1

### Relational analysis result of NS_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0931715, upper bound: 3295.0928177
time: 0.74 seconds

## Relational analysis of NS_B1_A2_B1_B1_A2

### Relational analysis result of NS_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0931715, upper bound: 3295.0928177
time: 0.95 seconds

## BFS NS instance: NS_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -7290.9477539, 13141.9960938, -6640.5722656, 12021.3955078, -19312.3437500, 19782.5625000
1: -1416.7792969, 1863.9151611, -1300.3208008, 1706.5837402, -3123.3630371, 3164.2358398
2: -1050.0471191, 2161.3654785, -960.2090454, 1979.3630371, -3029.4091797, 3121.5744629
3: -1125.0623779, 2893.8081055, -1027.2272949, 2642.2919922, -3767.3542480, 3921.0354004
4: -895.8900146, 2696.9577637, -818.7729492, 2470.3601074, -3366.2500000, 3515.7297363

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A2_B1_B2_A1

### Relational analysis result of NS_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0931717, upper bound: 3295.0928177
time: 0.88 seconds

## Relational analysis of NS_B1_A2_B1_B2_A2

### Relational analysis result of NS_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0931717, upper bound: 3295.0928177
time: 0.81 seconds

## BFS NS instance: NS_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7211.5590820, 12997.2832031, -7867.9946289, 14280.7128906, -21492.2675781, 20865.2675781
1: -1401.1241455, 1842.7766113, -1543.8864746, 2019.1888428, -3420.3129883, 3386.6630859
2: -1038.5899658, 2138.8242188, -1138.9371338, 2348.4260254, -3387.0156250, 3277.7612305
3: -1113.2752686, 2862.4819336, -1218.2897949, 3138.2448730, -4251.5200195, 4080.7712402
4: -886.3181152, 2668.6738281, -972.5296021, 2927.4938965, -3813.8112793, 3641.2028809

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B2_A1_B1

### Relational analysis result of NS_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0898563, upper bound: 3295.0908872
time: 0.87 seconds

## Relational analysis of NS_B1_A2_B2_A1_B2

### Relational analysis result of NS_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0898563, upper bound: 3295.0908873
time: 0.85 seconds

## BFS NS instance: NS_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7373.1738281, 13304.9658203, -7889.7900391, 14318.2441406, -21691.4179688, 21194.7539062
1: -1437.5773926, 1887.0266113, -1547.8332520, 2024.8664551, -3462.4438477, 3434.8598633
2: -1063.0716553, 2188.3273926, -1142.0759277, 2354.8752441, -3417.9467773, 3330.4030762
3: -1138.7753906, 2927.7038574, -1221.4556885, 3146.6604004, -4285.4355469, 4149.1591797
4: -907.2762451, 2728.9245605, -975.1660156, 2935.5219727, -3842.7983398, 3704.0905762

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B2_A2_B1

### Relational analysis result of NS_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0898564, upper bound: 3295.0908872
time: 0.81 seconds

## Relational analysis of NS_B1_A2_B2_A2_B2

### Relational analysis result of NS_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0898564, upper bound: 3295.0908872
time: 0.88 seconds

## BFS NS instance: NS_B2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -6483.1650391, 11737.7138672, -7290.8300781, 13142.9208984, -19626.0820312, 19028.5429688
1: -1269.7789307, 1667.1958008, -1417.0169678, 1863.8448486, -3133.6237793, 3084.2128906
2: -937.7656860, 1932.8701172, -1050.0134277, 2161.5070801, -3099.2724609, 2982.8833008
3: -1002.7440186, 2579.8486328, -1125.1123047, 2893.9077148, -3896.6511230, 3704.9609375
4: -799.5012817, 2412.3791504, -895.8804321, 2697.0876465, -3496.5883789, 3308.2595215

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A1_A1_A1_B1

### Relational analysis result of NS_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0928177, upper bound: 3295.0931716
time: 0.87 seconds

## Relational analysis of NS_B2_A1_A1_A1_B2

### Relational analysis result of NS_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0928177, upper bound: 3295.0932504
time: 0.79 seconds

## BFS NS instance: NS_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -6640.5722656, 12021.3955078, -7290.9477539, 13141.9960938, -19782.5625000, 19312.3437500
1: -1300.3208008, 1706.5837402, -1416.7792969, 1863.9151611, -3164.2358398, 3123.3630371
2: -960.2090454, 1979.3630371, -1050.0471191, 2161.3654785, -3121.5744629, 3029.4091797
3: -1027.2272949, 2642.2919922, -1125.0623779, 2893.8081055, -3921.0354004, 3767.3542480
4: -818.7729492, 2470.3601074, -895.8900146, 2696.9577637, -3515.7294922, 3366.2500000

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A1_A1_A2_B1

### Relational analysis result of NS_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0928177, upper bound: 3295.0931717
time: 0.80 seconds

## Relational analysis of NS_B2_A1_A1_A2_B2

### Relational analysis result of NS_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0928177, upper bound: 3295.0932540
time: 0.79 seconds

## BFS NS instance: NS_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -7867.9946289, 14280.7128906, -7211.5590820, 12997.2832031, -20865.2675781, 21492.2675781
1: -1543.8864746, 2019.1888428, -1401.1241455, 1842.7766113, -3386.6630859, 3420.3129883
2: -1138.9371338, 2348.4260254, -1038.5899658, 2138.8242188, -3277.7612305, 3387.0156250
3: -1218.2897949, 3138.2448730, -1113.2752686, 2862.4819336, -4080.7712402, 4251.5200195
4: -972.5296021, 2927.4938965, -886.3181152, 2668.6738281, -3641.2031250, 3813.8112793

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B2_A1_A2_B1_A1

### Relational analysis result of NS_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0908873, upper bound: 3295.0898562
time: 0.85 seconds

## Relational analysis of NS_B2_A1_A2_B1_A2

### Relational analysis result of NS_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0908873, upper bound: 3295.0898564
time: 0.85 seconds

## BFS NS instance: NS_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -7889.7900391, 14318.2441406, -7373.1738281, 13304.9658203, -21194.7539062, 21691.4179688
1: -1547.8332520, 2024.8664551, -1437.5773926, 1887.0266113, -3434.8596191, 3462.4438477
2: -1142.0759277, 2354.8752441, -1063.0716553, 2188.3273926, -3330.4030762, 3417.9467773
3: -1221.4556885, 3146.6604004, -1138.7753906, 2927.7038574, -4149.1596680, 4285.4355469
4: -975.1660156, 2935.5219727, -907.2762451, 2728.9245605, -3704.0905762, 3842.7983398

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B2_A1_A2_B2_A1

### Relational analysis result of NS_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0908873, upper bound: 3295.0898563
time: 0.78 seconds

## Relational analysis of NS_B2_A1_A2_B2_A2

### Relational analysis result of NS_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0908873, upper bound: 3295.0898563
time: 0.81 seconds

## BFS NS instance: NS_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7283.7900391, 13129.8017578, -7283.7900391, 13129.8017578, -20413.5898438, 20413.5898438
1: -1415.9500732, 1862.2276611, -1415.9500732, 1862.2276611, -3278.1777344, 3278.1777344
2: -1049.0753174, 2159.5200195, -1049.0753174, 2159.5200195, -3208.5949707, 3208.5949707
3: -1124.2354736, 2890.9475098, -1124.2354736, 2890.9475098, -4015.1828613, 4015.1828613
4: -895.0813599, 2694.6889648, -895.0813599, 2694.6889648, -3589.7702637, 3589.7702637

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B2_A2_B1_A1_B1

### Relational analysis result of NS_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0932474, upper bound: 3295.0933150
time: 0.85 seconds

## Relational analysis of NS_B2_A2_B1_A1_B2

### Relational analysis result of NS_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0932521, upper bound: 3295.0932958
time: 0.87 seconds

## BFS NS instance: NS_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7364.7934570, 13272.5234375, -7283.7900391, 13129.8017578, -20494.5937500, 20556.3105469
1: -1431.2687988, 1882.0081787, -1415.9500732, 1862.2276611, -3293.4965820, 3297.9582520
2: -1060.3488770, 2182.7976074, -1049.0753174, 2159.5200195, -3219.8686523, 3231.8730469
3: -1136.6070557, 2922.7785645, -1124.2354736, 2890.9475098, -4027.5544434, 4047.0139160
4: -904.7531128, 2723.8620605, -895.0813599, 2694.6889648, -3599.4421387, 3618.9431152

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_B1_A2_B1

### Relational analysis result of NS_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0910645, upper bound: 3295.0923725
time: 0.90 seconds

## Relational analysis of NS_B2_A2_B1_A2_B2

### Relational analysis result of NS_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0910629, upper bound: 3295.0910584
time: 0.92 seconds

## BFS NS instance: NS_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7283.7900391, 13129.8017578, -7364.7934570, 13272.5234375, -20556.3105469, 20494.5917969
1: -1415.9500732, 1862.2276611, -1431.2687988, 1882.0081787, -3297.9580078, 3293.4965820
2: -1049.0753174, 2159.5200195, -1060.3488770, 2182.7976074, -3231.8730469, 3219.8686523
3: -1124.2354736, 2890.9475098, -1136.6070557, 2922.7785645, -4047.0139160, 4027.5544434
4: -895.0813599, 2694.6889648, -904.7531128, 2723.8620605, -3618.9431152, 3599.4421387

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_B2_A1_A1

### Relational analysis result of NS_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0924259, upper bound: 3295.0910570
time: 0.87 seconds

## Relational analysis of NS_B2_A2_B2_A1_A2

### Relational analysis result of NS_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0910587, upper bound: 3295.0910565
time: 0.87 seconds

## BFS NS instance: NS_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7364.7934570, 13272.5234375, -7364.7934570, 13272.5234375, -20637.3125000, 20637.3125000
1: -1431.2687988, 1882.0081787, -1431.2687988, 1882.0081787, -3313.2768555, 3313.2768555
2: -1060.3488770, 2182.7976074, -1060.3488770, 2182.7976074, -3243.1464844, 3243.1464844
3: -1136.6070557, 2922.7785645, -1136.6070557, 2922.7785645, -4059.3854980, 4059.3854980
4: -904.7531128, 2723.8620605, -904.7531128, 2723.8620605, -3628.6149902, 3628.6149902

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B2_A2_B2_A2_B1

### Relational analysis result of NS_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0932468, upper bound: 3295.0934761
time: 0.89 seconds

## Relational analysis of NS_B2_A2_B2_A2_B2

### Relational analysis result of NS_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0932494, upper bound: 3295.0932632
time: 0.86 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.75 seconds
NS_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.75
Output dim: 3, lower bound: -3295.0816298, upper bound: 3295.0871327
NS_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.75
Output dim: 3, lower bound: -3295.0816256, upper bound: 3295.0816249
NS_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.75
Output dim: 3, lower bound: -3295.0911344, upper bound: 3295.0925121
NS_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.75
Output dim: 3, lower bound: -3295.0911366, upper bound: 3295.0913249
NS_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.75
Output dim: 3, lower bound: -3295.0923728, upper bound: 3295.0911339
NS_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.75
Output dim: 3, lower bound: -3295.0911353, upper bound: 3295.0911345
NS_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.75
Output dim: 3, lower bound: -3295.0869548, upper bound: 3295.0857480
NS_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.75
Output dim: 3, lower bound: -3295.0816250, upper bound: 3295.0857498
NS_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.75
Output dim: 3, lower bound: -3295.0931715, upper bound: 3295.0928177
NS_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.75
Output dim: 3, lower bound: -3295.0931715, upper bound: 3295.0928177
NS_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.75
Output dim: 3, lower bound: -3295.0931717, upper bound: 3295.0928177
NS_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.75
Output dim: 3, lower bound: -3295.0931717, upper bound: 3295.0928177
NS_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.75
Output dim: 3, lower bound: -3295.0898563, upper bound: 3295.0908872
NS_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.75
Output dim: 3, lower bound: -3295.0898563, upper bound: 3295.0908873
NS_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.75
Output dim: 3, lower bound: -3295.0898564, upper bound: 3295.0908872
NS_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.75
Output dim: 3, lower bound: -3295.0898564, upper bound: 3295.0908872
NS_B2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.75
Output dim: 3, lower bound: -3295.0928177, upper bound: 3295.0931716
NS_B2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.75
Output dim: 3, lower bound: -3295.0928177, upper bound: 3295.0932504
NS_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.75
Output dim: 3, lower bound: -3295.0928177, upper bound: 3295.0931717
NS_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.75
Output dim: 3, lower bound: -3295.0928177, upper bound: 3295.0932540
NS_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.75
Output dim: 3, lower bound: -3295.0908873, upper bound: 3295.0898562
NS_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.75
Output dim: 3, lower bound: -3295.0908873, upper bound: 3295.0898564
NS_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.75
Output dim: 3, lower bound: -3295.0908873, upper bound: 3295.0898563
NS_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.75
Output dim: 3, lower bound: -3295.0908873, upper bound: 3295.0898563
NS_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.75
Output dim: 3, lower bound: -3295.0932474, upper bound: 3295.0933150
NS_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.75
Output dim: 3, lower bound: -3295.0932521, upper bound: 3295.0932958
NS_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.75
Output dim: 3, lower bound: -3295.0910645, upper bound: 3295.0923725
NS_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.75
Output dim: 3, lower bound: -3295.0910629, upper bound: 3295.0910584
NS_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.75
Output dim: 3, lower bound: -3295.0924259, upper bound: 3295.0910570
NS_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.75
Output dim: 3, lower bound: -3295.0910587, upper bound: 3295.0910565
NS_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.75
Output dim: 3, lower bound: -3295.0932468, upper bound: 3295.0934761
NS_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.75
Output dim: 3, lower bound: -3295.0932494, upper bound: 3295.0932632

## BFS NS instance: NS_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -6489.8256836, 11756.0917969, -6416.4311523, 11626.5791016, -18116.4023438, 18172.5175781
1: -1272.1302490, 1669.7869873, -1258.2335205, 1651.2559814, -2923.3854980, 2928.0205078
2: -939.0661621, 1936.0194092, -928.5156250, 1914.7067871, -2853.7729492, 2864.5351562
3: -1004.2418213, 2583.3149414, -992.9560547, 2554.5605469, -3558.8022461, 3576.2709961
4: -800.7533569, 2415.9438477, -791.8477783, 2389.1821289, -3189.9355469, 3207.7915039

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0816255, upper bound: 3295.0816254
time: 0.77 seconds

## Relational analysis of NS_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0816255, upper bound: 3295.0816256
time: 0.80 seconds

## BFS NS instance: NS_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -6525.1860352, 11815.4531250, -7210.8886719, 13047.1660156, -19572.3515625, 19026.3378906
1: -1278.4985352, 1678.3087158, -1412.7145996, 1852.8453369, -3131.3435059, 3091.0227051
2: -944.0284424, 1945.1094971, -1040.5878906, 2147.1069336, -3091.1352539, 2985.6972656
3: -1009.5937500, 2596.7285156, -1113.1927490, 2866.8137207, -3876.4074707, 3709.9213867
4: -804.8950806, 2427.4843750, -887.7133789, 2676.2360840, -3481.1311035, 3315.1977539

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0816257, upper bound: 3295.0816255
time: 0.80 seconds

## Relational analysis of NS_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0816257, upper bound: 3295.0816256
time: 0.84 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -6640.0649414, 12021.5429688, -6439.3149414, 11659.4384766, -18299.5039062, 18460.8554688
1: -1300.7160645, 1706.2124023, -1261.6285400, 1655.6556396, -2956.3713379, 2967.8400879
2: -960.2606812, 1978.8305664, -931.5660400, 1919.1628418, -2879.4235840, 2910.3964844
3: -1027.8200684, 2641.4733887, -996.8132324, 2561.1254883, -3588.9455566, 3638.2866211
4: -819.0267944, 2469.4370117, -794.4699097, 2394.9016113, -3213.9284668, 3263.9064941

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0911344, upper bound: 3295.0913248
time: 0.81 seconds

## Relational analysis of NS_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0911344, upper bound: 3295.0913249
time: 0.88 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -6687.0664062, 12105.6679688, -6775.2822266, 12254.9990234, -18942.0605469, 18880.9472656
1: -1309.6104736, 1719.1092529, -1326.7280273, 1741.8665771, -3051.4768066, 3045.8369141
2: -967.3370972, 1993.8002930, -979.8616333, 2019.5083008, -2986.8454590, 2973.6618652
3: -1035.0112305, 2661.4204102, -1049.0399170, 2693.5725098, -3728.5834961, 3710.4602051
4: -824.9645386, 2488.3754883, -835.9738770, 2519.1958008, -3344.1601562, 3324.3491211

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0911366, upper bound: 3295.0913249
time: 0.77 seconds

## Relational analysis of NS_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0911366, upper bound: 3295.0913249
time: 0.86 seconds

## BFS NS instance: NS_B1_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -6439.3149414, 11659.4384766, -6640.0649414, 12021.5429688, -18460.8554688, 18299.5039062
1: -1261.6285400, 1655.6556396, -1300.7160645, 1706.2124023, -2967.8400879, 2956.3713379
2: -931.5660400, 1919.1628418, -960.2606812, 1978.8305664, -2910.3964844, 2879.4235840
3: -996.8132324, 2561.1254883, -1027.8200684, 2641.4733887, -3638.2866211, 3588.9455566
4: -794.4699097, 2394.9016113, -819.0267944, 2469.4370117, -3263.9064941, 3213.9284668

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_B2_A1_A1_B1

### Relational analysis result of NS_B1_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0913249, upper bound: 3295.0911345
time: 0.90 seconds

## Relational analysis of NS_B1_A1_B2_A1_A1_B2

### Relational analysis result of NS_B1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0913249, upper bound: 3295.0911344
time: 0.88 seconds

## BFS NS instance: NS_B1_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -6775.2822266, 12254.9990234, -6687.0664062, 12105.6679688, -18880.9472656, 18942.0625000
1: -1326.7280273, 1741.8665771, -1309.6104736, 1719.1092529, -3045.8369141, 3051.4768066
2: -979.8616333, 2019.5083008, -967.3370972, 1993.8002930, -2973.6618652, 2986.8454590
3: -1049.0399170, 2693.5725098, -1035.0112305, 2661.4204102, -3710.4602051, 3728.5834961
4: -835.9738770, 2519.1958008, -824.9645386, 2488.3754883, -3324.3491211, 3344.1601562

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_B2_A1_A2_B1

### Relational analysis result of NS_B1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0913249, upper bound: 3295.0911365
time: 0.86 seconds

## Relational analysis of NS_B1_A1_B2_A1_A2_B2

### Relational analysis result of NS_B1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0913249, upper bound: 3295.0911366
time: 0.82 seconds

## BFS NS instance: NS_B1_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -6573.5454102, 11908.3896484, -6647.6611328, 12039.9707031, -18613.5156250, 18556.0507812
1: -1288.6230469, 1690.3634033, -1302.7761230, 1709.1724854, -2997.7954102, 2993.1396484
2: -950.8310547, 1960.9499512, -961.5466309, 1982.5086670, -2933.3398438, 2922.4965820
3: -1017.3301392, 2616.5153809, -1028.7788086, 2645.7124023, -3663.0424805, 3645.2941895
4: -811.0277710, 2446.8420410, -820.0775146, 2473.9274902, -3284.9550781, 3266.9191895

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_B2_A2_A1_B1

### Relational analysis result of NS_B1_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0857527, upper bound: 3295.0857479
time: 0.87 seconds

## Relational analysis of NS_B1_A1_B2_A2_A1_B2

### Relational analysis result of NS_B1_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0857527, upper bound: 3295.0857478
time: 0.90 seconds

## BFS NS instance: NS_B1_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -7333.4741211, 13261.0087891, -6679.9497070, 12094.9316406, -19428.4023438, 19940.9570312
1: -1436.2659912, 1883.2271729, -1308.7176514, 1716.9831543, -3153.2490234, 3191.9448242
2: -1058.0214844, 2183.1591797, -966.1133423, 1990.9440918, -3048.9653320, 3149.2722168
3: -1132.5695801, 2913.7424316, -1033.6785889, 2658.1010742, -3790.6706543, 3947.4208984
4: -902.6582642, 2722.3146973, -823.8541260, 2484.7224121, -3387.3803711, 3546.1689453

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_B2_A2_A2_B1

### Relational analysis result of NS_B1_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0857530, upper bound: 3295.0857498
time: 0.82 seconds

## Relational analysis of NS_B1_A1_B2_A2_A2_B2

### Relational analysis result of NS_B1_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0857530, upper bound: 3295.0857498
time: 0.86 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -7247.3491211, 13062.9619141, -6483.1650391, 11737.7138672, -18985.0625000, 19546.1230469
1: -1408.5805664, 1852.6773682, -1269.7789307, 1667.1958008, -3075.7763672, 3122.4562988
2: -1043.6920166, 2148.5249023, -937.7656860, 1932.8701172, -2976.5620117, 3086.2900391
3: -1118.4425049, 2876.3554688, -1002.7440186, 2579.8486328, -3698.2907715, 3879.0988770
4: -890.4755859, 2680.9982910, -799.5012817, 2412.3791504, -3302.8547363, 3480.4990234

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B1_B1_A1_B1

### Relational analysis result of NS_B1_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0868462, upper bound: 3295.0828008
time: 0.87 seconds

## Relational analysis of NS_B1_A2_B1_B1_A1_B2

### Relational analysis result of NS_B1_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0931716, upper bound: 3295.0928602
time: 1.11 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -7327.9013672, 13204.4814453, -6483.1650391, 11737.7138672, -19065.6152344, 19687.6464844
1: -1423.7720947, 1872.3315430, -1269.7789307, 1667.1958008, -3090.9677734, 3142.1103516
2: -1054.8836670, 2171.6694336, -937.7656860, 1932.8701172, -2987.7539062, 3109.4345703
3: -1130.7302246, 2907.9482422, -1002.7440186, 2579.8486328, -3710.5786133, 3910.6914062
4: -900.0734253, 2710.0100098, -799.5012817, 2412.3791504, -3312.4526367, 3509.5107422

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B1_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0868462, upper bound: 3295.0828009
time: 0.80 seconds

## Relational analysis of NS_B1_A2_B1_B1_A2_B2

### Relational analysis result of NS_B1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0931715, upper bound: 3295.0928603
time: 0.81 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -7247.3491211, 13062.9619141, -6640.5722656, 12021.3955078, -19268.7441406, 19703.5273438
1: -1408.5805664, 1852.6773682, -1300.3208008, 1706.5837402, -3115.1643066, 3152.9975586
2: -1043.6920166, 2148.5249023, -960.2090454, 1979.3630371, -3023.0546875, 3108.7338867
3: -1118.4425049, 2876.3554688, -1027.2272949, 2642.2919922, -3760.7341309, 3903.5827637
4: -890.4755859, 2680.9982910, -818.7729492, 2470.3601074, -3360.8354492, 3499.7707520

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B1_B2_A1_B1

### Relational analysis result of NS_B1_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0868459, upper bound: 3295.0828107
time: 0.85 seconds

## Relational analysis of NS_B1_A2_B1_B2_A1_B2

### Relational analysis result of NS_B1_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0931715, upper bound: 3295.0928177
time: 1.06 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -7327.9013672, 13204.4814453, -6640.5722656, 12021.3955078, -19349.2968750, 19845.0507812
1: -1423.7720947, 1872.3315430, -1300.3208008, 1706.5837402, -3130.3559570, 3172.6518555
2: -1054.8836670, 2171.6694336, -960.2090454, 1979.3630371, -3034.2463379, 3131.8784180
3: -1130.7302246, 2907.9482422, -1027.2272949, 2642.2919922, -3773.0219727, 3935.1752930
4: -900.0734253, 2710.0100098, -818.7729492, 2470.3601074, -3370.4335938, 3528.7824707

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B1_B2_A2_B1

### Relational analysis result of NS_B1_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0868459, upper bound: 3295.0828102
time: 0.91 seconds

## Relational analysis of NS_B1_A2_B1_B2_A2_B2

### Relational analysis result of NS_B1_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0931715, upper bound: 3295.0928177
time: 0.84 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -7211.5590820, 12997.2832031, -7834.0898438, 14219.3076172, -21430.8632812, 20831.3632812
1: -1401.1241455, 1842.7766113, -1537.2507324, 2010.3123779, -3411.4362793, 3380.0273438
2: -1038.5899658, 2138.8242188, -1133.9804688, 2338.2800293, -3376.8696289, 3272.8046875
3: -1113.2752686, 2862.4819336, -1213.0609131, 3124.7866211, -4238.0620117, 4075.5429688
4: -886.3181152, 2668.6738281, -968.3118896, 2914.8735352, -3801.1911621, 3636.9851074

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0914461, upper bound: 3295.0906392
time: 0.84 seconds

## Relational analysis of NS_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0899640, upper bound: 3295.0897788
time: 0.78 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -7211.5590820, 12997.2832031, -8078.5366211, 14668.9267578, -21880.4804688, 21075.8085938
1: -1401.1241455, 1842.7766113, -1587.5946045, 2075.0661621, -3476.1904297, 3430.3710938
2: -1038.5899658, 2138.8242188, -1169.8382568, 2410.9777832, -3449.5673828, 3308.6618652
3: -1113.2752686, 2862.4819336, -1251.8041992, 3220.8073730, -4334.0825195, 4114.2861328
4: -886.3181152, 2668.6738281, -999.2598877, 3003.6501465, -3889.9680176, 3667.9331055

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0917991, upper bound: 3295.0910411
time: 0.81 seconds

## Relational analysis of NS_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0907682, upper bound: 3295.0910524
time: 0.89 seconds

## BFS NS instance: NS_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -7373.1738281, 13304.9658203, -7834.0898438, 14219.3076172, -21592.4804688, 21139.0546875
1: -1437.5773926, 1887.0266113, -1537.2507324, 2010.3123779, -3447.8896484, 3424.2773438
2: -1063.0716553, 2188.3273926, -1133.9804688, 2338.2800293, -3401.3515625, 3322.3078613
3: -1138.7753906, 2927.7038574, -1213.0609131, 3124.7866211, -4263.5620117, 4140.7646484
4: -907.2762451, 2728.9245605, -968.3118896, 2914.8735352, -3822.1496582, 3697.2363281

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0898541, upper bound: 3295.0908872
time: 0.84 seconds

## Relational analysis of NS_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0898541, upper bound: 3295.0908872
time: 0.90 seconds

## BFS NS instance: NS_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -7373.1738281, 13304.9658203, -8078.5366211, 14668.9267578, -22042.0976562, 21383.4980469
1: -1437.5773926, 1887.0266113, -1587.5946045, 2075.0661621, -3512.6435547, 3474.6206055
2: -1063.0716553, 2188.3273926, -1169.8382568, 2410.9777832, -3474.0490723, 3358.1647949
3: -1138.7753906, 2927.7038574, -1251.8041992, 3220.8073730, -4359.5830078, 4179.5078125
4: -907.2762451, 2728.9245605, -999.2598877, 3003.6501465, -3910.9262695, 3728.1843262

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0898455, upper bound: 3295.0893582
time: 0.82 seconds

## Relational analysis of NS_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0875638, upper bound: 3295.0892882
time: 0.74 seconds

## BFS NS instance: NS_B2_A1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -6483.1650391, 11737.7138672, -7247.3491211, 13062.9619141, -19546.1250000, 18985.0625000
1: -1269.7789307, 1667.1958008, -1408.5805664, 1852.6773682, -3122.4562988, 3075.7763672
2: -937.7656860, 1932.8701172, -1043.6920166, 2148.5249023, -3086.2900391, 2976.5620117
3: -1002.7440186, 2579.8486328, -1118.4425049, 2876.3554688, -3879.0988770, 3698.2907715
4: -799.5012817, 2412.3791504, -890.4755859, 2680.9982910, -3480.4990234, 3302.8547363

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_A1_A1_A1_B1_A1

### Relational analysis result of NS_B2_A1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0828009, upper bound: 3295.0868462
time: 0.85 seconds

## Relational analysis of NS_B2_A1_A1_A1_B1_A2

### Relational analysis result of NS_B2_A1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0928602, upper bound: 3295.0931715
time: 0.98 seconds

## BFS NS instance: NS_B2_A1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -6483.1650391, 11737.7138672, -7327.9013672, 13204.4814453, -19687.6464844, 19065.6152344
1: -1269.7789307, 1667.1958008, -1423.7720947, 1872.3315430, -3142.1103516, 3090.9677734
2: -937.7656860, 1932.8701172, -1054.8836670, 2171.6694336, -3109.4345703, 2987.7539062
3: -1002.7440186, 2579.8486328, -1130.7302246, 2907.9482422, -3910.6914062, 3710.5786133
4: -799.5012817, 2412.3791504, -900.0734253, 2710.0100098, -3509.5107422, 3312.4526367

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_A1_A1_A1_B2_A1

### Relational analysis result of NS_B2_A1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0828009, upper bound: 3295.0868462
time: 0.82 seconds

## Relational analysis of NS_B2_A1_A1_A1_B2_A2

### Relational analysis result of NS_B2_A1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0928603, upper bound: 3295.0932503
time: 0.89 seconds

## BFS NS instance: NS_B2_A1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -6640.5722656, 12021.3955078, -7247.3491211, 13062.9619141, -19703.5273438, 19268.7441406
1: -1300.3208008, 1706.5837402, -1408.5805664, 1852.6773682, -3152.9975586, 3115.1643066
2: -960.2090454, 1979.3630371, -1043.6920166, 2148.5249023, -3108.7338867, 3023.0546875
3: -1027.2272949, 2642.2919922, -1118.4425049, 2876.3554688, -3903.5827637, 3760.7341309
4: -818.7729492, 2470.3601074, -890.4755859, 2680.9982910, -3499.7707520, 3360.8354492

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_A1_A1_A2_B1_A1

### Relational analysis result of NS_B2_A1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0828107, upper bound: 3295.0868459
time: 0.93 seconds

## Relational analysis of NS_B2_A1_A1_A2_B1_A2

### Relational analysis result of NS_B2_A1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0928177, upper bound: 3295.0931716
time: 0.85 seconds

## BFS NS instance: NS_B2_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -6640.5722656, 12021.3955078, -7327.9013672, 13204.4814453, -19845.0507812, 19349.2968750
1: -1300.3208008, 1706.5837402, -1423.7720947, 1872.3315430, -3172.6518555, 3130.3559570
2: -960.2090454, 1979.3630371, -1054.8836670, 2171.6694336, -3131.8784180, 3034.2463379
3: -1027.2272949, 2642.2919922, -1130.7302246, 2907.9482422, -3935.1752930, 3773.0219727
4: -818.7729492, 2470.3601074, -900.0734253, 2710.0100098, -3528.7824707, 3370.4335938

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_A1_A1_A2_B2_A1

### Relational analysis result of NS_B2_A1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0828107, upper bound: 3295.0868459
time: 0.98 seconds

## Relational analysis of NS_B2_A1_A1_A2_B2_A2

### Relational analysis result of NS_B2_A1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0928177, upper bound: 3295.0932540
time: 0.89 seconds

## BFS NS instance: NS_B2_A1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7834.0898438, 14219.3076172, -7211.5590820, 12997.2832031, -20831.3632812, 21430.8632812
1: -1537.2507324, 2010.3123779, -1401.1241455, 1842.7766113, -3380.0273438, 3411.4362793
2: -1133.9804688, 2338.2800293, -1038.5899658, 2138.8242188, -3272.8046875, 3376.8696289
3: -1213.0609131, 3124.7866211, -1113.2752686, 2862.4819336, -4075.5429688, 4238.0620117
4: -968.3118896, 2914.8735352, -886.3181152, 2668.6738281, -3636.9851074, 3801.1911621

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B2_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_A2_B1_A1_B1

### Relational analysis result of NS_B2_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0906391, upper bound: 3295.0914458
time: 0.87 seconds

## Relational analysis of NS_B2_A1_A2_B1_A1_B2

### Relational analysis result of NS_B2_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0897789, upper bound: 3295.0899638
time: 0.80 seconds

## BFS NS instance: NS_B2_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8078.5366211, 14668.9267578, -7211.5590820, 12997.2832031, -21075.8085938, 21880.4804688
1: -1587.5946045, 2075.0661621, -1401.1241455, 1842.7766113, -3430.3710938, 3476.1904297
2: -1169.8382568, 2410.9777832, -1038.5899658, 2138.8242188, -3308.6618652, 3449.5673828
3: -1251.8041992, 3220.8073730, -1113.2752686, 2862.4819336, -4114.2861328, 4334.0825195
4: -999.2598877, 3003.6501465, -886.3181152, 2668.6738281, -3667.9331055, 3889.9682617

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B2_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_A1_A2_B1_A2_B1

### Relational analysis result of NS_B2_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0910411, upper bound: 3295.0920627
time: 0.89 seconds

## Relational analysis of NS_B2_A1_A2_B1_A2_B2

### Relational analysis result of NS_B2_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0910523, upper bound: 3295.0907681
time: 0.85 seconds

## BFS NS instance: NS_B2_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7834.0898438, 14219.3076172, -7373.1738281, 13304.9658203, -21139.0546875, 21592.4804688
1: -1537.2507324, 2010.3123779, -1437.5773926, 1887.0266113, -3424.2773438, 3447.8896484
2: -1133.9804688, 2338.2800293, -1063.0716553, 2188.3273926, -3322.3078613, 3401.3515625
3: -1213.0609131, 3124.7866211, -1138.7753906, 2927.7038574, -4140.7646484, 4263.5620117
4: -968.3118896, 2914.8735352, -907.2762451, 2728.9245605, -3697.2363281, 3822.1496582

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B2_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B2_A1_A2_B2_A1_B1

### Relational analysis result of NS_B2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0908873, upper bound: 3295.0898563
time: 0.84 seconds

## Relational analysis of NS_B2_A1_A2_B2_A1_B2

### Relational analysis result of NS_B2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0908872, upper bound: 3295.0898564
time: 0.84 seconds

## BFS NS instance: NS_B2_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8078.5366211, 14668.9267578, -7373.1738281, 13304.9658203, -21383.4980469, 22042.0976562
1: -1587.5946045, 2075.0661621, -1437.5773926, 1887.0266113, -3474.6206055, 3512.6435547
2: -1169.8382568, 2410.9777832, -1063.0716553, 2188.3273926, -3358.1650391, 3474.0490723
3: -1251.8041992, 3220.8073730, -1138.7753906, 2927.7038574, -4179.5078125, 4359.5830078
4: -999.2598877, 3003.6501465, -907.2762451, 2728.9245605, -3728.1843262, 3910.9262695

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B2_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_A1_A2_B2_A2_B1

### Relational analysis result of NS_B2_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0893582, upper bound: 3295.0898455
time: 0.83 seconds

## Relational analysis of NS_B2_A1_A2_B2_A2_B2

### Relational analysis result of NS_B2_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0892884, upper bound: 3295.0875639
time: 0.84 seconds

## BFS NS instance: NS_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -7162.9204102, 12909.4316406, -7115.3232422, 12818.1044922, -19981.0253906, 20024.7539062
1: -1392.8815918, 1831.2720947, -1383.1793213, 1818.8621826, -3211.7436523, 3214.4514160
2: -1031.5617676, 2123.0822754, -1024.5582275, 2108.2753906, -3139.8371582, 3147.6406250
3: -1105.5543213, 2842.0732422, -1098.1826172, 2822.1770020, -3927.7309570, 3940.2558594
4: -880.2127686, 2648.7324219, -874.2543945, 2630.0971680, -3510.3095703, 3522.9863281

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0933018, upper bound: 3295.0933015
time: 0.95 seconds

## Relational analysis of NS_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0933019, upper bound: 3295.0933067
time: 0.88 seconds

## BFS NS instance: NS_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -7249.2856445, 13067.9882812, -7228.2602539, 13030.3417969, -20279.6269531, 20296.2460938
1: -1409.4114990, 1853.4055176, -1405.4310303, 1848.0343018, -3257.4453125, 3258.8356934
2: -1044.1199951, 2149.1398926, -1041.1029053, 2142.8227539, -3186.9416504, 3190.2426758
3: -1118.9339600, 2877.2004395, -1115.7045898, 2868.8273926, -3987.7612305, 3992.9050293
4: -890.8538818, 2681.7031250, -888.2798462, 2673.7980957, -3564.6518555, 3569.9829102

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0933072, upper bound: 3295.0933013
time: 0.90 seconds

## Relational analysis of NS_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0933072, upper bound: 3295.0933071
time: 0.85 seconds

## BFS NS instance: NS_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7277.0312500, 13111.3134766, -7146.7666016, 12878.6787109, -20155.7070312, 20258.0800781
1: -1414.1300049, 1859.1104736, -1389.2469482, 1826.5540771, -3240.6840820, 3248.3574219
2: -1047.4138184, 2156.2202148, -1028.9301758, 2118.0053711, -3165.4187012, 3185.1503906
3: -1122.8641357, 2887.3962402, -1102.8461914, 2835.7490234, -3958.6130371, 3990.2424316
4: -893.7340698, 2690.5471191, -877.9409180, 2642.6582031, -3536.3923340, 3568.4875488

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0910629, upper bound: 3295.0910584
time: 0.85 seconds

## Relational analysis of NS_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0910629, upper bound: 3295.0910584
time: 0.89 seconds

## BFS NS instance: NS_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7251.5776367, 13065.1259766, -7439.6118164, 13412.5703125, -20664.1484375, 20504.7382812
1: -1407.9389648, 1852.7208252, -1445.8487549, 1903.6831055, -3311.6220703, 3298.5695801
2: -1043.8315430, 2148.5729980, -1072.0802002, 2205.7299805, -3249.5612793, 3220.6533203
3: -1118.8834229, 2876.9956055, -1148.4760742, 2951.3193359, -4070.2026367, 4025.4716797
4: -890.6506348, 2681.1418457, -914.7959595, 2751.1452637, -3641.7954102, 3595.9377441

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0910628, upper bound: 3295.0910584
time: 0.81 seconds

## Relational analysis of NS_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0910629, upper bound: 3295.0910584
time: 0.88 seconds

## BFS NS instance: NS_B2_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -7146.7666016, 12878.6787109, -7277.0312500, 13111.3134766, -20258.0800781, 20155.7050781
1: -1389.2469482, 1826.5540771, -1414.1300049, 1859.1104736, -3248.3574219, 3240.6840820
2: -1028.9301758, 2118.0053711, -1047.4138184, 2156.2202148, -3185.1503906, 3165.4187012
3: -1102.8461914, 2835.7490234, -1122.8641357, 2887.3962402, -3990.2424316, 3958.6132812
4: -877.9409180, 2642.6582031, -893.7340698, 2690.5471191, -3568.4875488, 3536.3920898

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_B2_A1_A1_B1

### Relational analysis result of NS_B2_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0910603, upper bound: 3295.0910625
time: 0.84 seconds

## Relational analysis of NS_B2_A2_B2_A1_A1_B2

### Relational analysis result of NS_B2_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0910603, upper bound: 3295.0910625
time: 0.83 seconds

## BFS NS instance: NS_B2_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -7439.6118164, 13412.5703125, -7251.5776367, 13065.1259766, -20504.7382812, 20664.1484375
1: -1445.8487549, 1903.6831055, -1407.9389648, 1852.7208252, -3298.5695801, 3311.6220703
2: -1072.0802002, 2205.7299805, -1043.8315430, 2148.5729980, -3220.6533203, 3249.5610352
3: -1148.4760742, 2951.3193359, -1118.8834229, 2876.9956055, -4025.4716797, 4070.2026367
4: -914.7959595, 2751.1452637, -890.6506348, 2681.1418457, -3595.9377441, 3641.7954102

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_B2_A1_A2_B1

### Relational analysis result of NS_B2_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0910603, upper bound: 3295.0910625
time: 0.90 seconds

## Relational analysis of NS_B2_A2_B2_A1_A2_B2

### Relational analysis result of NS_B2_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0910602, upper bound: 3295.0910625
time: 0.84 seconds

## BFS NS instance: NS_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -7243.2802734, 13051.1103516, -7199.6010742, 12967.6835938, -20210.9628906, 20250.7089844
1: -1408.0567627, 1850.8588867, -1399.2109375, 1839.5913086, -3247.6479492, 3250.0698242
2: -1042.7269287, 2146.1794434, -1036.3757324, 2132.4873047, -3175.2136230, 3182.5546875
3: -1117.7854004, 2873.6787109, -1111.1197510, 2855.3786621, -3973.1640625, 3984.7983398
4: -889.7750244, 2677.7001953, -884.3880615, 2660.4401855, -3550.2148438, 3562.0883789

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0932646, upper bound: 3295.0932626
time: 0.95 seconds

## Relational analysis of NS_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0932645, upper bound: 3295.0932632
time: 0.81 seconds

## BFS NS instance: NS_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -7331.3710938, 13212.7324219, -7311.1557617, 13176.5742188, -20507.9433594, 20523.8847656
1: -1424.9583740, 1873.4498291, -1421.1423340, 1868.2738037, -3293.2321777, 3294.5922852
2: -1055.5438232, 2172.7702637, -1052.6372070, 2166.7045898, -3222.2480469, 3225.4074707
3: -1131.4758301, 2909.4892578, -1128.3723145, 2901.4509277, -4032.9265137, 4037.8613281
4: -900.6580200, 2711.3093262, -898.1812744, 2703.7145996, -3604.3725586, 3609.4904785

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0932680, upper bound: 3295.0932625
time: 0.95 seconds

## Relational analysis of NS_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0932679, upper bound: 3295.0932631
time: 0.81 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.35 seconds
NS_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0816255, upper bound: 3295.0816254
NS_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0816255, upper bound: 3295.0816256
NS_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0816257, upper bound: 3295.0816255
NS_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0816257, upper bound: 3295.0816256
NS_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0911344, upper bound: 3295.0913248
NS_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0911344, upper bound: 3295.0913249
NS_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0911366, upper bound: 3295.0913249
NS_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0911366, upper bound: 3295.0913249
NS_B1_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0913249, upper bound: 3295.0911345
NS_B1_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0913249, upper bound: 3295.0911344
NS_B1_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0913249, upper bound: 3295.0911365
NS_B1_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0913249, upper bound: 3295.0911366
NS_B1_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0857527, upper bound: 3295.0857479
NS_B1_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0857527, upper bound: 3295.0857478
NS_B1_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0857530, upper bound: 3295.0857498
NS_B1_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0857530, upper bound: 3295.0857498
NS_B1_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0868462, upper bound: 3295.0828008
NS_B1_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0931716, upper bound: 3295.0928602
NS_B1_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0868462, upper bound: 3295.0828009
NS_B1_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0931715, upper bound: 3295.0928603
NS_B1_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0868459, upper bound: 3295.0828107
NS_B1_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0931715, upper bound: 3295.0928177
NS_B1_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0868459, upper bound: 3295.0828102
NS_B1_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0931715, upper bound: 3295.0928177
NS_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0914461, upper bound: 3295.0906392
NS_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0899640, upper bound: 3295.0897788
NS_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0917991, upper bound: 3295.0910411
NS_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0907682, upper bound: 3295.0910524
NS_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0898541, upper bound: 3295.0908872
NS_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0898541, upper bound: 3295.0908872
NS_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0898455, upper bound: 3295.0893582
NS_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0875638, upper bound: 3295.0892882
NS_B2_A1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0828009, upper bound: 3295.0868462
NS_B2_A1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0928602, upper bound: 3295.0931715
NS_B2_A1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0828009, upper bound: 3295.0868462
NS_B2_A1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0928603, upper bound: 3295.0932503
NS_B2_A1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0828107, upper bound: 3295.0868459
NS_B2_A1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0928177, upper bound: 3295.0931716
NS_B2_A1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0828107, upper bound: 3295.0868459
NS_B2_A1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0928177, upper bound: 3295.0932540
NS_B2_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0906391, upper bound: 3295.0914458
NS_B2_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0897789, upper bound: 3295.0899638
NS_B2_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0910411, upper bound: 3295.0920627
NS_B2_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0910523, upper bound: 3295.0907681
NS_B2_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0908873, upper bound: 3295.0898563
NS_B2_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0908872, upper bound: 3295.0898564
NS_B2_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0893582, upper bound: 3295.0898455
NS_B2_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0892884, upper bound: 3295.0875639
NS_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0933018, upper bound: 3295.0933015
NS_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0933019, upper bound: 3295.0933067
NS_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0933072, upper bound: 3295.0933013
NS_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0933072, upper bound: 3295.0933071
NS_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0910629, upper bound: 3295.0910584
NS_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0910629, upper bound: 3295.0910584
NS_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0910628, upper bound: 3295.0910584
NS_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0910629, upper bound: 3295.0910584
NS_B2_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0910603, upper bound: 3295.0910625
NS_B2_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0910603, upper bound: 3295.0910625
NS_B2_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0910603, upper bound: 3295.0910625
NS_B2_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0910602, upper bound: 3295.0910625
NS_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0932646, upper bound: 3295.0932626
NS_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0932645, upper bound: 3295.0932632
NS_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0932680, upper bound: 3295.0932625
NS_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 3, lower bound: -3295.0932679, upper bound: 3295.0932631

## BFS NS instance: NS_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -6416.4311523, 11626.5791016, -6416.4311523, 11626.5791016, -18043.0097656, 18043.0078125
1: -1258.2335205, 1651.2559814, -1258.2335205, 1651.2559814, -2909.4895020, 2909.4895020
2: -928.5156250, 1914.7067871, -928.5156250, 1914.7067871, -2843.2224121, 2843.2224121
3: -992.9560547, 2554.5605469, -992.9560547, 2554.5605469, -3547.5166016, 3547.5166016
4: -791.8477783, 2389.1821289, -791.8477783, 2389.1821289, -3181.0297852, 3181.0297852

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B1_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_A1_B1_A1_B1_A1_A1

### Relational analysis result of NS_B1_A1_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0815914, upper bound: 3295.0871092
time: 0.79 seconds

## Relational analysis of NS_B1_A1_B1_A1_B1_A1_A2

### Relational analysis result of NS_B1_A1_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0816298, upper bound: 3295.0870386
time: 0.93 seconds

## BFS NS instance: NS_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7210.8886719, 13047.1660156, -6416.4311523, 11626.5791016, -18837.4667969, 19463.5917969
1: -1412.7145996, 1852.8453369, -1258.2335205, 1651.2559814, -3063.9699707, 3111.0788574
2: -1040.5878906, 2147.1069336, -928.5156250, 1914.7067871, -2955.2946777, 3075.6225586
3: -1113.1927490, 2866.8137207, -992.9560547, 2554.5605469, -3667.7534180, 3859.7697754
4: -887.7133789, 2676.2360840, -791.8477783, 2389.1821289, -3276.8955078, 3468.0839844

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0813731, upper bound: 3295.0863292
time: 0.87 seconds

## Relational analysis of NS_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0813782, upper bound: 3295.0863291
time: 0.90 seconds

## BFS NS instance: NS_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6416.4311523, 11626.5791016, -7210.8886719, 13047.1660156, -19463.5937500, 18837.4667969
1: -1258.2335205, 1651.2559814, -1412.7145996, 1852.8453369, -3111.0788574, 3063.9699707
2: -928.5156250, 1914.7067871, -1040.5878906, 2147.1069336, -3075.6225586, 2955.2946777
3: -992.9560547, 2554.5605469, -1113.1927490, 2866.8137207, -3859.7697754, 3667.7534180
4: -791.8477783, 2389.1821289, -887.7133789, 2676.2360840, -3468.0839844, 3276.8955078

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B1_A1_B1_A1_B2_A1_A1

### Relational analysis result of NS_B1_A1_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0816128, upper bound: 3295.0813724
time: 0.83 seconds

## Relational analysis of NS_B1_A1_B1_A1_B2_A1_A2

### Relational analysis result of NS_B1_A1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0813748, upper bound: 3295.0813748
time: 0.78 seconds

## BFS NS instance: NS_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7210.8886719, 13047.1660156, -7210.8886719, 13047.1660156, -20258.0488281, 20258.0527344
1: -1412.7145996, 1852.8453369, -1412.7145996, 1852.8453369, -3265.5598145, 3265.5598145
2: -1040.5878906, 2147.1069336, -1040.5878906, 2147.1069336, -3187.6948242, 3187.6948242
3: -1113.1927490, 2866.8137207, -1113.1927490, 2866.8137207, -3980.0063477, 3980.0063477
4: -887.7133789, 2676.2360840, -887.7133789, 2676.2360840, -3563.9494629, 3563.9494629

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_A1_B1_A1_B2_A2_A1

### Relational analysis result of NS_B1_A1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0815875, upper bound: 3295.0816196
time: 0.95 seconds

## Relational analysis of NS_B1_A1_B1_A1_B2_A2_A2

### Relational analysis result of NS_B1_A1_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0816254, upper bound: 3295.0816255
time: 0.83 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -6599.9536133, 11949.0380859, -6439.3149414, 11659.4384766, -18259.3925781, 18388.3515625
1: -1292.8790283, 1695.7641602, -1261.6285400, 1655.6556396, -2948.5346680, 2957.3920898
2: -954.4169922, 1966.6712646, -931.5660400, 1919.1628418, -2873.5798340, 2898.2373047
3: -1021.6961670, 2625.5429688, -996.8132324, 2561.1254883, -3582.8212891, 3622.3562012
4: -814.0932617, 2454.1315918, -794.4699097, 2394.9016113, -3208.9946289, 3248.6015625

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0911230, upper bound: 3295.0924419
time: 0.84 seconds

## Relational analysis of NS_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0911202, upper bound: 3295.0923181
time: 0.85 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -6948.3916016, 12566.2978516, -6439.3149414, 11659.4384766, -18607.8300781, 19005.6132812
1: -1360.7685547, 1785.3769531, -1261.6285400, 1655.6556396, -3016.4243164, 3047.0048828
2: -1004.6387939, 2070.6440430, -931.5660400, 1919.1628418, -2923.8017578, 3002.2099609
3: -1076.0729980, 2762.5310059, -996.8132324, 2561.1254883, -3637.1982422, 3759.3437500
4: -857.2727661, 2582.9453125, -794.4699097, 2394.9016113, -3252.1743164, 3377.4147949

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0850502, upper bound: 3295.0821329
time: 0.88 seconds

## Relational analysis of NS_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0911344, upper bound: 3295.0925121
time: 0.91 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -6599.9536133, 11949.0380859, -6775.2822266, 12254.9990234, -18854.9531250, 18724.3183594
1: -1292.8790283, 1695.7641602, -1326.7280273, 1741.8665771, -3034.7456055, 3022.4919434
2: -954.4169922, 1966.6712646, -979.8616333, 2019.5083008, -2973.9252930, 2946.5329590
3: -1021.6961670, 2625.5429688, -1049.0399170, 2693.5725098, -3715.2685547, 3674.5830078
4: -814.0932617, 2454.1315918, -835.9738770, 2519.1958008, -3333.2883301, 3290.1054688

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_B1_A2_B2_A1_A1

### Relational analysis result of NS_B1_A1_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0828222, upper bound: 3295.0809237
time: 0.85 seconds

## Relational analysis of NS_B1_A1_B1_A2_B2_A1_A2

### Relational analysis result of NS_B1_A1_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0812750, upper bound: 3295.0809107
time: 0.81 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6948.3916016, 12566.2978516, -6775.2822266, 12254.9990234, -19203.3906250, 19341.5800781
1: -1360.7685547, 1785.3769531, -1326.7280273, 1741.8665771, -3102.6352539, 3112.1047363
2: -1004.6387939, 2070.6440430, -979.8616333, 2019.5083008, -3024.1469727, 3050.5056152
3: -1076.0729980, 2762.5310059, -1049.0399170, 2693.5725098, -3769.6452637, 3811.5705566
4: -857.2727661, 2582.9453125, -835.9738770, 2519.1958008, -3376.4685059, 3418.9189453

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0911230, upper bound: 3295.0913249
time: 0.84 seconds

## Relational analysis of NS_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0911202, upper bound: 3295.0913203
time: 0.88 seconds

## BFS NS instance: NS_B1_A1_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -6439.3149414, 11659.4384766, -6599.9536133, 11949.0380859, -18388.3496094, 18259.3925781
1: -1261.6285400, 1655.6556396, -1292.8790283, 1695.7641602, -2957.3918457, 2948.5346680
2: -931.5660400, 1919.1628418, -954.4169922, 1966.6712646, -2898.2373047, 2873.5798340
3: -996.8132324, 2561.1254883, -1021.6961670, 2625.5429688, -3622.3562012, 3582.8212891
4: -794.4699097, 2394.9016113, -814.0932617, 2454.1315918, -3248.6015625, 3208.9946289

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B1_A1_B2_A1_A1_B1_A1

### Relational analysis result of NS_B1_A1_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0924419, upper bound: 3295.0911229
time: 0.83 seconds

## Relational analysis of NS_B1_A1_B2_A1_A1_B1_A2

### Relational analysis result of NS_B1_A1_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0923181, upper bound: 3295.0911200
time: 0.82 seconds

## BFS NS instance: NS_B1_A1_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -6439.3149414, 11659.4384766, -6948.3916016, 12566.2978516, -19005.6132812, 18607.8300781
1: -1261.6285400, 1655.6556396, -1360.7685547, 1785.3769531, -3047.0048828, 3016.4243164
2: -931.5660400, 1919.1628418, -1004.6387939, 2070.6440430, -3002.2099609, 2923.8017578
3: -996.8132324, 2561.1254883, -1076.0729980, 2762.5310059, -3759.3437500, 3637.1982422
4: -794.4699097, 2394.9016113, -857.2727661, 2582.9453125, -3377.4147949, 3252.1743164

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B1_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_A1_B2_A1_A1_B2_A1

### Relational analysis result of NS_B1_A1_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0821331, upper bound: 3295.0850502
time: 0.87 seconds

## Relational analysis of NS_B1_A1_B2_A1_A1_B2_A2

### Relational analysis result of NS_B1_A1_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0821329, upper bound: 3295.0911344
time: 0.83 seconds

## BFS NS instance: NS_B1_A1_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -6775.2822266, 12254.9990234, -6599.9536133, 11949.0380859, -18724.3183594, 18854.9531250
1: -1326.7280273, 1741.8665771, -1292.8790283, 1695.7641602, -3022.4919434, 3034.7456055
2: -979.8616333, 2019.5083008, -954.4169922, 1966.6712646, -2946.5329590, 2973.9252930
3: -1049.0399170, 2693.5725098, -1021.6961670, 2625.5429688, -3674.5827637, 3715.2685547
4: -835.9738770, 2519.1958008, -814.0932617, 2454.1315918, -3290.1054688, 3333.2883301

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_B2_A1_A2_B1_B1

### Relational analysis result of NS_B1_A1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0809237, upper bound: 3295.0855455
time: 0.79 seconds

## Relational analysis of NS_B1_A1_B2_A1_A2_B1_B2

### Relational analysis result of NS_B1_A1_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0809106, upper bound: 3295.0827047
time: 0.88 seconds

## BFS NS instance: NS_B1_A1_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -6775.2822266, 12254.9990234, -6948.3916016, 12566.2978516, -19341.5800781, 19203.3906250
1: -1326.7280273, 1741.8665771, -1360.7685547, 1785.3769531, -3112.1047363, 3102.6352539
2: -979.8616333, 2019.5083008, -1004.6387939, 2070.6440430, -3050.5056152, 3024.1469727
3: -1049.0399170, 2693.5725098, -1076.0729980, 2762.5310059, -3811.5708008, 3769.6452637
4: -835.9738770, 2519.1958008, -857.2727661, 2582.9453125, -3418.9189453, 3376.4685059

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B1_A1_B2_A1_A2_B2_A1

### Relational analysis result of NS_B1_A1_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0913249, upper bound: 3295.0911266
time: 0.81 seconds

## Relational analysis of NS_B1_A1_B2_A1_A2_B2_A2

### Relational analysis result of NS_B1_A1_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0913204, upper bound: 3295.0911244
time: 0.80 seconds

## BFS NS instance: NS_B1_A1_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -6573.5454102, 11908.3896484, -6573.5454102, 11908.3896484, -18481.9335938, 18481.9335938
1: -1288.6230469, 1690.3634033, -1288.6230469, 1690.3634033, -2978.9863281, 2978.9863281
2: -950.8310547, 1960.9499512, -950.8310547, 1960.9499512, -2911.7810059, 2911.7810059
3: -1017.3301392, 2616.5153809, -1017.3301392, 2616.5153809, -3633.8454590, 3633.8454590
4: -811.0277710, 2446.8420410, -811.0277710, 2446.8420410, -3257.8696289, 3257.8696289

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B1_A1_B2_A2_A1_B1_A1

### Relational analysis result of NS_B1_A1_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0889742, upper bound: 3295.0857440
time: 0.86 seconds

## Relational analysis of NS_B1_A1_B2_A2_A1_B1_A2

### Relational analysis result of NS_B1_A1_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0889742, upper bound: 3295.0857470
time: 0.83 seconds

## BFS NS instance: NS_B1_A1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -6573.5454102, 11908.3896484, -7333.4741211, 13261.0087891, -19834.5546875, 19241.8593750
1: -1288.6230469, 1690.3634033, -1436.2659912, 1883.2271729, -3171.8500977, 3126.6293945
2: -950.8310547, 1960.9499512, -1058.0214844, 2183.1591797, -3133.9902344, 3018.9711914
3: -1017.3301392, 2616.5153809, -1132.5695801, 2913.7424316, -3931.0725098, 3749.0849609
4: -811.0277710, 2446.8420410, -902.6582642, 2722.3146973, -3533.3425293, 3349.5000000

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B1_A1_B2_A2_A1_B2_A1

### Relational analysis result of NS_B1_A1_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0881642, upper bound: 3295.0856985
time: 0.90 seconds

## Relational analysis of NS_B1_A1_B2_A2_A1_B2_A2

### Relational analysis result of NS_B1_A1_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0881642, upper bound: 3295.0857085
time: 0.84 seconds

## BFS NS instance: NS_B1_A1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -7333.4741211, 13261.0087891, -6573.5454102, 11908.3896484, -19241.8593750, 19834.5546875
1: -1436.2659912, 1883.2271729, -1288.6230469, 1690.3634033, -3126.6293945, 3171.8500977
2: -1058.0214844, 2183.1591797, -950.8310547, 1960.9499512, -3018.9711914, 3133.9902344
3: -1132.5695801, 2913.7424316, -1017.3301392, 2616.5153809, -3749.0849609, 3931.0725098
4: -902.6582642, 2722.3146973, -811.0277710, 2446.8420410, -3349.5000000, 3533.3425293

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_A1_B2_A2_A2_B1_B1

### Relational analysis result of NS_B1_A1_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0857076, upper bound: 3295.0857485
time: 0.84 seconds

## Relational analysis of NS_B1_A1_B2_A2_A2_B1_B2

### Relational analysis result of NS_B1_A1_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0857128, upper bound: 3295.0857094
time: 1.02 seconds

## BFS NS instance: NS_B1_A1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -7333.4741211, 13261.0087891, -7333.4741211, 13261.0087891, -20594.4804688, 20594.4804688
1: -1436.2659912, 1883.2271729, -1436.2659912, 1883.2271729, -3319.4931641, 3319.4931641
2: -1058.0214844, 2183.1591797, -1058.0214844, 2183.1591797, -3241.1799316, 3241.1801758
3: -1132.5695801, 2913.7424316, -1132.5695801, 2913.7424316, -4046.3120117, 4046.3120117
4: -902.6582642, 2722.3146973, -902.6582642, 2722.3146973, -3624.9729004, 3624.9729004

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B1_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_A1_B2_A2_A2_B2_A1

### Relational analysis result of NS_B1_A1_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0855018, upper bound: 3295.0855828
time: 0.86 seconds

## Relational analysis of NS_B1_A1_B2_A2_A2_B2_A2

### Relational analysis result of NS_B1_A1_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0816233, upper bound: 3295.0855896
time: 0.89 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -7199.9848633, 12973.5732422, -6312.2001953, 11421.1250000, -18621.1093750, 19285.7734375
1: -1398.7991943, 1840.5628662, -1235.2764893, 1623.6853027, -3022.4843750, 3075.8391113
2: -1036.7811279, 2134.0368652, -912.9312134, 1880.6274414, -2917.4079590, 3046.9680176
3: -1111.1219482, 2856.9406738, -976.3247681, 2510.7509766, -3621.8728027, 3833.2653809
4: -884.5511475, 2662.9250488, -778.2130127, 2347.2919922, -3231.8425293, 3441.1376953

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B1_B1_A1_B1_B1

### Relational analysis result of NS_B1_A2_B1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0868521, upper bound: 3295.0815967
time: 0.91 seconds

## Relational analysis of NS_B1_A2_B1_B1_A1_B1_B2

### Relational analysis result of NS_B1_A2_B1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0870160, upper bound: 3295.0828022
time: 0.82 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -7247.3491211, 13062.9619141, -6480.5820312, 11733.0166016, -18980.3652344, 19543.5410156
1: -1408.5805664, 1852.6773682, -1269.2646484, 1666.5177002, -3075.0981445, 3121.9416504
2: -1043.6920166, 2148.5249023, -937.3853760, 1932.0944824, -2975.7866211, 3085.9101562
3: -1118.4425049, 2876.3554688, -1002.3391724, 2578.8281250, -3697.2702637, 3878.6943359
4: -890.4755859, 2680.9982910, -799.1769409, 2411.4204102, -3301.8959961, 3480.1748047

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B1_B1_A1_B2_A1

### Relational analysis result of NS_B1_A2_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0827132, upper bound: 3295.0871873
time: 0.93 seconds

## Relational analysis of NS_B1_A2_B1_B1_A1_B2_A2

### Relational analysis result of NS_B1_A2_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0827132, upper bound: 3295.0928674
time: 0.84 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7279.3251953, 13112.8798828, -6312.2001953, 11421.1250000, -18700.4492188, 19425.0800781
1: -1413.7623291, 1859.9237061, -1235.2764893, 1623.6853027, -3037.4477539, 3095.1999512
2: -1047.7988281, 2156.8356934, -912.9312134, 1880.6274414, -2928.4257812, 3069.7668457
3: -1123.2264404, 2888.0468750, -976.3247681, 2510.7509766, -3633.9770508, 3864.3715820
4: -893.9992065, 2691.5231934, -778.2130127, 2347.2919922, -3241.2912598, 3469.7358398

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B1_B1_A2_B1_B1

### Relational analysis result of NS_B1_A2_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0818825, upper bound: 3295.0801837
time: 0.80 seconds

## Relational analysis of NS_B1_A2_B1_B1_A2_B1_B2

### Relational analysis result of NS_B1_A2_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0818862, upper bound: 3295.0793241
time: 0.84 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7327.9013672, 13204.4814453, -6480.5820312, 11733.0166016, -19060.9140625, 19685.0625000
1: -1423.7720947, 1872.3315430, -1269.2646484, 1666.5177002, -3090.2897949, 3141.5959473
2: -1054.8836670, 2171.6694336, -937.3853760, 1932.0944824, -2986.9780273, 3109.0546875
3: -1130.7302246, 2907.9482422, -1002.3391724, 2578.8281250, -3709.5581055, 3910.2868652
4: -900.0734253, 2710.0100098, -799.1769409, 2411.4204102, -3311.4938965, 3509.1865234

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B1_B1_A2_B2_A1

### Relational analysis result of NS_B1_A2_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826420, upper bound: 3295.0871797
time: 1.05 seconds

## Relational analysis of NS_B1_A2_B1_B1_A2_B2_A2

### Relational analysis result of NS_B1_A2_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826082, upper bound: 3295.0928603
time: 0.78 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -7199.9848633, 12973.5732422, -6467.6103516, 11700.7734375, -18900.7558594, 19441.1816406
1: -1398.7991943, 1840.5628662, -1265.4416504, 1662.5065918, -3061.3051758, 3106.0043945
2: -1036.7811279, 2134.0368652, -935.0703125, 1926.5522461, -2963.3334961, 3069.1071777
3: -1111.1219482, 2856.9406738, -1000.5393677, 2572.2927246, -3683.4145508, 3857.4799805
4: -884.5511475, 2662.9250488, -797.2370605, 2404.5961914, -3289.1472168, 3460.1621094

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B1_A2_B1_B2_A1_B1_B1

### Relational analysis result of NS_B1_A2_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0867143, upper bound: 3295.0827376
time: 0.87 seconds

## Relational analysis of NS_B1_A2_B1_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_A2_B1_B2_A1_B1_B1

### Relational analysis result of NS_B1_A2_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826879, upper bound: 3295.0809886
time: 0.82 seconds

## Relational analysis of NS_B1_A2_B1_B2_A1_B1_B2

### Relational analysis result of NS_B1_A2_B1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826785, upper bound: 3295.0790242
time: 0.81 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -7247.3491211, 13062.9619141, -6638.1953125, 12017.0996094, -19264.4492188, 19701.1503906
1: -1408.5805664, 1852.6773682, -1299.8514404, 1705.9632568, -3114.5437012, 3152.5283203
2: -1043.6920166, 2148.5249023, -959.8612671, 1978.6480713, -3022.3398438, 3108.3862305
3: -1118.4425049, 2876.3554688, -1026.8560791, 2641.3571777, -3759.7995605, 3903.2114258
4: -890.4755859, 2680.9982910, -818.4764404, 2469.4760742, -3359.9514160, 3499.4741211

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B1_B2_A1_B2_A1

### Relational analysis result of NS_B1_A2_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0827132, upper bound: 3295.0870858
time: 1.01 seconds

## Relational analysis of NS_B1_A2_B1_B2_A1_B2_A2

### Relational analysis result of NS_B1_A2_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0827132, upper bound: 3295.0928223
time: 0.89 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -7279.3251953, 13112.8798828, -6467.6103516, 11700.7734375, -18980.0976562, 19580.4882812
1: -1413.7623291, 1859.9237061, -1265.4416504, 1662.5065918, -3076.2687988, 3125.3652344
2: -1047.7988281, 2156.8356934, -935.0703125, 1926.5522461, -2974.3510742, 3091.9060059
3: -1123.2264404, 2888.0468750, -1000.5393677, 2572.2927246, -3695.5187988, 3888.5861816
4: -893.9992065, 2691.5231934, -797.2370605, 2404.5961914, -3298.5954590, 3488.7602539

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B1_A2_B1_B2_A2_B1_B1

### Relational analysis result of NS_B1_A2_B1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0867061, upper bound: 3295.0827328
time: 0.90 seconds

## Relational analysis of NS_B1_A2_B1_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B1_B2_A2_B1_B1

### Relational analysis result of NS_B1_A2_B1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0818825, upper bound: 3295.0801837
time: 0.87 seconds

## Relational analysis of NS_B1_A2_B1_B2_A2_B1_B2

### Relational analysis result of NS_B1_A2_B1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0818853, upper bound: 3295.0793242
time: 0.81 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -7327.9013672, 13204.4814453, -6638.1953125, 12017.0996094, -19345.0000000, 19842.6738281
1: -1423.7720947, 1872.3315430, -1299.8514404, 1705.9632568, -3129.7348633, 3172.1826172
2: -1054.8836670, 2171.6694336, -959.8612671, 1978.6480713, -3033.5314941, 3131.5307617
3: -1130.7302246, 2907.9482422, -1026.8560791, 2641.3571777, -3772.0874023, 3934.8041992
4: -900.0734253, 2710.0100098, -818.4764404, 2469.4760742, -3369.5495605, 3528.4858398

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B1_B2_A2_B2_A1

### Relational analysis result of NS_B1_A2_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826420, upper bound: 3295.0870815
time: 0.85 seconds

## Relational analysis of NS_B1_A2_B1_B2_A2_B2_A2

### Relational analysis result of NS_B1_A2_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0826082, upper bound: 3295.0928177
time: 0.94 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -7076.0468750, 12747.7832031, -7742.0708008, 14053.3105469, -21126.8613281, 20489.8535156
1: -1375.1789551, 1807.8424072, -1519.4791260, 1986.1350098, -3361.3132324, 3327.3215332
2: -1018.9938354, 2098.2414551, -1120.6138916, 2310.3813477, -3329.1010742, 3218.8554688
3: -1092.4702148, 2807.9838867, -1199.1219482, 3087.6303711, -4180.0048828, 4007.1047363
4: -869.7048340, 2617.8193359, -957.0497437, 2879.9565430, -3749.6611328, 3574.8691406

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0914162, upper bound: 3295.0913313
time: 0.86 seconds

## Relational analysis of NS_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0914161, upper bound: 3295.0913313
time: 0.87 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7325.1757812, 13201.4033203, -7785.2011719, 14127.8925781, -21453.0683594, 20986.6054688
1: -1423.8216553, 1872.3203125, -1527.3303223, 1997.6676025, -3421.4882812, 3399.6506348
2: -1055.2849121, 2173.5310059, -1126.9099121, 2323.7116699, -3378.9965820, 3300.4406738
3: -1131.5407715, 2906.9414062, -1205.6359863, 3105.3754883, -4236.9155273, 4112.5771484
4: -900.9309692, 2711.0270996, -962.3711548, 2896.7104492, -3797.6411133, 3673.3979492

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0914162, upper bound: 3295.0913330
time: 0.86 seconds

## Relational analysis of NS_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0914162, upper bound: 3295.0913329
time: 0.89 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -7067.4721680, 12733.1064453, -8019.8574219, 14562.6347656, -21623.9003906, 20752.9589844
1: -1372.9982910, 1805.2589111, -1576.2562256, 2059.5590820, -3432.5573730, 3381.5151367
2: -1017.3977051, 2095.2048340, -1161.2105713, 2392.9289551, -3410.3259277, 3256.4152832
3: -1090.7529297, 2804.4335938, -1242.7019043, 3197.3730469, -4288.0454102, 4047.1352539
4: -868.2774048, 2614.0559082, -991.9583130, 2980.9450684, -3849.2224121, 3606.0141602

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0903048, upper bound: 3295.0910411
time: 0.82 seconds

## Relational analysis of NS_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0903048, upper bound: 3295.0910411
time: 0.82 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7323.7709961, 13201.6777344, -7912.8798828, 14366.7363281, -21690.5058594, 21114.5566406
1: -1422.6414795, 1872.8713379, -1554.1578369, 2032.8087158, -3455.4501953, 3427.0292969
2: -1055.1584473, 2172.1477051, -1145.7919922, 2361.4716797, -3416.6296387, 3317.9396973
3: -1130.5831299, 2905.7109375, -1225.7741699, 3154.5834961, -4285.1665039, 4131.4848633
4: -900.4917603, 2709.2363281, -978.6367798, 2941.9814453, -3842.4731445, 3687.8730469

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0903048, upper bound: 3295.0910523
time: 0.85 seconds

## Relational analysis of NS_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -3295.0903048, upper bound: 3295.0910523
time: 0.83 seconds

## BFS NS instance: NS_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7355.9868164, 13274.0039062, -7834.0898438, 14219.3076172, -21575.2929688, 21108.0937500
1: -1433.9339600, 1882.2539062, -1537.2507324, 2010.3123779, -3444.2463379, 3419.5046387
2: -1060.1401367, 2181.8471680, -1133.9804688, 2338.2800293, -3398.4201660, 3315.8276367
3: -1135.3851318, 2920.9599609, -1213.0609131, 3124.7866211, -4260.1718750, 4134.0209961
4: -904.6696167, 2720.9099121, -968.3118896, 2914.8735352, -3819.5429688, 3689.2216797

Time for backsubstitution: 2.46 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.54 + 416.92 = 421.46 seconds
