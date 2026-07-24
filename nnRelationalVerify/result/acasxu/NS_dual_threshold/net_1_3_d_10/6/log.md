## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_3.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 6)
Time budget: 420 seconds
Split limit: 100
Threshold: 19342.684212278527


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125)
1: (-12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062)
2: (-18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625)
3: (-6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750)
4: (-20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.62 + 2.19 = 2.82 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -19343.0710737, upper bound: 19343.0710737

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0537087, upper bound: 19343.0219288
time: 0.86 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0560443, upper bound: 19343.0560440
time: 0.90 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.83 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.83
Output dim: 3, lower bound: -19343.0537087, upper bound: 19343.0219288
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.83
Output dim: 3, lower bound: -19343.0560443, upper bound: 19343.0560440

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -15757.4970703, 11657.4091797, -15979.9384766, 11824.5693359, -27582.0664062, 27637.3476562
1: -12754.1386719, 11285.7158203, -12933.5390625, 11445.0302734, -24199.1660156, 24219.2519531
2: -18603.0605469, 12328.0322266, -18861.8515625, 12502.3134766, -31105.3750000, 31189.8828125
3: -6435.5620117, 17520.4550781, -6526.2768555, 17762.8984375, -24198.4609375, 24046.7285156
4: -20442.2402344, 12188.4228516, -20728.6542969, 12361.6982422, -32803.9375000, 32917.0781250

Time for backsubstitution: 0.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0184050, upper bound: 19343.0184050
time: 0.88 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0184050, upper bound: 19343.0219288
time: 0.94 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -17640.9335938, 13213.5449219, -15848.3613281, 11744.4130859, -29385.3437500, 29061.9062500
1: -14277.3007812, 12790.0429688, -12825.9814453, 11367.3994141, -25644.6992188, 25616.0234375
2: -20877.5839844, 13955.0917969, -18715.2031250, 12417.9443359, -33295.5234375, 32670.2929688
3: -7241.5898438, 19713.9003906, -6480.2543945, 17632.6621094, -24874.2500000, 26194.1542969
4: -22939.9179688, 13770.2119141, -20565.3750000, 12277.2470703, -35217.1640625, 34335.5820312

Time for backsubstitution: 0.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9496749, upper bound: 19343.0438296
time: 0.98 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0493307, upper bound: 19343.0493304
time: 0.95 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.54 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.54
Output dim: 3, lower bound: -19343.0184050, upper bound: 19343.0184050
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.54
Output dim: 3, lower bound: -19343.0184050, upper bound: 19343.0219288
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.54
Output dim: 3, lower bound: -19342.9496749, upper bound: 19343.0438296
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.54
Output dim: 3, lower bound: -19343.0493307, upper bound: 19343.0493304

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -15757.4970703, 11657.4091797, -15757.4970703, 11657.4091797, -27414.9062500, 27414.9062500
1: -12754.1386719, 11285.7158203, -12754.1386719, 11285.7158203, -24039.8515625, 24039.8515625
2: -18603.0605469, 12328.0322266, -18603.0605469, 12328.0322266, -30931.0917969, 30931.0898438
3: -6435.5620117, 17520.4550781, -6435.5620117, 17520.4550781, -23956.0175781, 23956.0156250
4: -20442.2402344, 12188.4228516, -20442.2402344, 12188.4228516, -32630.6542969, 32630.6542969

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9626120, upper bound: 19342.9617673
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9634526, upper bound: 19342.9634526
time: 0.81 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -15757.4970703, 11657.4091797, -17640.9335938, 13213.5449219, -28971.0429688, 29298.3437500
1: -12754.1386719, 11285.7158203, -14277.3007812, 12790.0429688, -25544.1816406, 25563.0156250
2: -18603.0605469, 12328.0322266, -20877.5839844, 13955.0917969, -32558.1484375, 33205.6132812
3: -6435.5620117, 17520.4550781, -7241.5898438, 19713.9003906, -26149.4609375, 24762.0429688
4: -20442.2402344, 12188.4228516, -22939.9179688, 13770.2119141, -34212.4531250, 35128.3359375

Time for backsubstitution: 0.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9626120, upper bound: 19342.9666274
time: 1.03 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9634526, upper bound: 19342.9683312
time: 0.77 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -17640.9335938, 13213.5449219, -15548.8027344, 11517.4150391, -29158.3476562, 28762.3476562
1: -14277.3007812, 12790.0429688, -12584.2441406, 11150.9130859, -25428.2128906, 25374.2871094
2: -20877.5839844, 13955.0917969, -18364.5878906, 12181.2207031, -33058.8046875, 32319.6796875
3: -7241.5898438, 19713.9003906, -6357.1464844, 17303.3007812, -24544.8886719, 26071.0468750
4: -22939.9179688, 13770.2119141, -20179.0683594, 12041.8300781, -34981.7500000, 33949.2812500

Time for backsubstitution: 0.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9496749, upper bound: 19343.0438298
time: 1.01 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9496749, upper bound: 19343.0438296
time: 0.75 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -17592.8066406, 13180.7851562, -17609.8437500, 13175.8964844, -30768.7031250, 30790.6250000
1: -14237.8349609, 12758.4804688, -14251.9882812, 12750.3837891, -26988.2187500, 27010.4687500
2: -20820.8906250, 13920.5976562, -20840.2285156, 13910.7500000, -34731.6406250, 34760.8281250
3: -7223.3847656, 19661.8359375, -7223.5888672, 19665.6523438, -26889.0371094, 26885.4257812
4: -22878.0957031, 13735.8193359, -22900.0292969, 13734.9853516, -36613.0820312, 36635.8476562

Time for backsubstitution: 0.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9683312, upper bound: 19343.0469735
time: 0.81 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9683312, upper bound: 19343.0493304
time: 0.77 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.20 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 3, lower bound: -19342.9626120, upper bound: 19342.9617673
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 3, lower bound: -19342.9634526, upper bound: 19342.9634526
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 3, lower bound: -19342.9626120, upper bound: 19342.9666274
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 3, lower bound: -19342.9634526, upper bound: 19342.9683312
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 3, lower bound: -19342.9496749, upper bound: 19343.0438298
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 3, lower bound: -19342.9496749, upper bound: 19343.0438296
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 3, lower bound: -19342.9683312, upper bound: 19343.0469735
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 3, lower bound: -19342.9683312, upper bound: 19343.0493304

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -15454.9707031, 11430.4335938, -15757.4970703, 11657.4091797, -27112.3789062, 27187.9296875
1: -12510.0507812, 11069.3183594, -12754.1386719, 11285.7158203, -23795.7656250, 23823.4570312
2: -18250.3164062, 12091.1337891, -18603.0605469, 12328.0322266, -30578.3476562, 30694.1953125
3: -6312.2260742, 17190.5605469, -6435.5620117, 17520.4550781, -23832.6796875, 23626.1210938
4: -20052.4492188, 11952.9658203, -20442.2402344, 12188.4228516, -32240.8691406, 32395.2011719

Time for backsubstitution: 0.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9609123, upper bound: 19342.9609123
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9609123, upper bound: 19342.9617673
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -17400.1269531, 13016.3066406, -15614.2412109, 11564.3574219, -28964.4843750, 28630.5429688
1: -14083.8212891, 12597.9062500, -12636.4814453, 11195.6855469, -25279.5078125, 25234.3867188
2: -20596.6406250, 13744.0898438, -18438.9316406, 12230.1435547, -32826.7851562, 32183.0175781
3: -7135.8437500, 19436.2500000, -6382.4208984, 17372.5273438, -24508.3710938, 25818.6699219
4: -22629.5566406, 13569.0703125, -20262.4628906, 12090.4218750, -34719.9765625, 33831.5312500

Time for backsubstitution: 0.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9617673, upper bound: 19342.9626120
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9617673, upper bound: 19342.9634526
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -15454.9707031, 11430.4335938, -17640.9335938, 13213.5449219, -28668.5156250, 29071.3632812
1: -12510.0507812, 11069.3183594, -14277.3007812, 12790.0429688, -25300.0937500, 25346.6191406
2: -18250.3164062, 12091.1337891, -20877.5839844, 13955.0917969, -32205.4082031, 32968.7187500
3: -6312.2260742, 17190.5605469, -7241.5898438, 19713.9003906, -26026.1269531, 24432.1464844
4: -20052.4492188, 11952.9658203, -22939.9179688, 13770.2119141, -33822.6601562, 34892.8828125

Time for backsubstitution: 0.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0445330, upper bound: 19342.9661731
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0392355, upper bound: 19342.9665842
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -17400.1269531, 13016.3066406, -17592.8066406, 13180.7851562, -30580.9101562, 30609.1113281
1: -14083.8212891, 12597.9062500, -14237.8349609, 12758.4804688, -26842.3007812, 26835.7421875
2: -20596.6406250, 13744.0898438, -20820.8906250, 13920.5976562, -34517.2265625, 34564.9804688
3: -7135.8437500, 19436.2500000, -7223.3847656, 19661.8359375, -26797.6796875, 26659.6347656
4: -22629.5566406, 13569.0703125, -22878.0957031, 13735.8193359, -36365.3750000, 36447.1640625

Time for backsubstitution: 0.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9464662, upper bound: 19342.9600414
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9464662, upper bound: 19342.9600414
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -17640.9335938, 13213.5449219, -15454.9707031, 11430.4335938, -29071.3652344, 28668.5156250
1: -14277.3007812, 12790.0429688, -12510.0507812, 11069.3183594, -25346.6191406, 25300.0937500
2: -20877.5839844, 13955.0917969, -18250.3164062, 12091.1337891, -32968.7148438, 32205.4062500
3: -7241.5898438, 19713.9003906, -6312.2260742, 17190.5605469, -24432.1464844, 26026.1269531
4: -22939.9179688, 13770.2119141, -20052.4492188, 11952.9658203, -34892.8828125, 33822.6601562

Time for backsubstitution: 0.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9465576, upper bound: 19343.0422491
time: 0.99 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9496749, upper bound: 19343.0369590
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -17640.9335938, 13213.5449219, -17336.7089844, 12989.7568359, -30630.6914062, 30550.2539062
1: -14277.3007812, 12790.0429688, -14032.3134766, 12576.0234375, -26853.3222656, 26822.3554688
2: -20877.5839844, 13955.0917969, -20523.5078125, 13721.3007812, -34598.8828125, 34478.5976562
3: -7241.5898438, 19713.9003906, -7118.8666992, 19383.9453125, -26625.5312500, 26832.7636719
4: -22939.9179688, 13770.2119141, -22548.2617188, 13537.5009766, -36477.4179688, 36318.4687500

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9465300, upper bound: 19342.9464961
time: 0.89 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9465300, upper bound: 19343.0438300
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: -17592.8066406, 13180.7851562, -17400.1269531, 13016.3066406, -30609.1132812, 30580.9101562
1: -14237.8349609, 12758.4804688, -14083.8212891, 12597.9062500, -26835.7421875, 26842.3007812
2: -20820.8906250, 13920.5976562, -20596.6406250, 13744.0898438, -34564.9804688, 34517.2304688
3: -7223.3847656, 19661.8359375, -7135.8437500, 19436.2500000, -26659.6347656, 26797.6796875
4: -22878.0957031, 13735.8193359, -22629.5566406, 13569.0703125, -36447.1640625, 36365.3750000

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9465300, upper bound: 19342.9464662
time: 0.86 seconds

## Relational analysis of NS_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9465300, upper bound: 19343.0469735
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: -17592.8066406, 13180.7851562, -19908.0585938, 15005.9785156, -32559.2031250, 33088.8437500
1: -14237.8349609, 12758.4804688, -16112.2021484, 14527.9570312, -28748.9296875, 28870.6835938
2: -20820.8906250, 13920.5976562, -23607.6992188, 15838.6308594, -36636.9687500, 37528.2968750
3: -7223.3847656, 19661.8359375, -8186.2548828, 22325.4218750, -29530.9472656, 27822.9589844
4: -22878.0957031, 13735.8193359, -25926.9453125, 15615.3916016, -38480.0898438, 39662.7656250

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9465300, upper bound: 19342.9464662
time: 0.80 seconds

## Relational analysis of NS_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9465300, upper bound: 19343.0493304
time: 0.69 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.13 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.13
Output dim: 3, lower bound: -19342.9609123, upper bound: 19342.9609123
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.13
Output dim: 3, lower bound: -19342.9609123, upper bound: 19342.9617673
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.13
Output dim: 3, lower bound: -19342.9617673, upper bound: 19342.9626120
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.13
Output dim: 3, lower bound: -19342.9617673, upper bound: 19342.9634526
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.13
Output dim: 3, lower bound: -19343.0445330, upper bound: 19342.9661731
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.13
Output dim: 3, lower bound: -19343.0392355, upper bound: 19342.9665842
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.13
Output dim: 3, lower bound: -19342.9464662, upper bound: 19342.9600414
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.13
Output dim: 3, lower bound: -19342.9464662, upper bound: 19342.9600414
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.13
Output dim: 3, lower bound: -19342.9465576, upper bound: 19343.0422491
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.13
Output dim: 3, lower bound: -19342.9496749, upper bound: 19343.0369590
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.13
Output dim: 3, lower bound: -19342.9465300, upper bound: 19342.9464961
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.13
Output dim: 3, lower bound: -19342.9465300, upper bound: 19343.0438300
NS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.13
Output dim: 3, lower bound: -19342.9465300, upper bound: 19342.9464662
NS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.13
Output dim: 3, lower bound: -19342.9465300, upper bound: 19343.0469735
NS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.13
Output dim: 3, lower bound: -19342.9465300, upper bound: 19342.9464662
NS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.13
Output dim: 3, lower bound: -19342.9465300, upper bound: 19343.0493304

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -15454.9707031, 11430.4335938, -15454.9707031, 11430.4335938, -26885.4042969, 26885.4042969
1: -12510.0507812, 11069.3183594, -12510.0507812, 11069.3183594, -23579.3691406, 23579.3691406
2: -18250.3164062, 12091.1337891, -18250.3164062, 12091.1337891, -30341.4492188, 30341.4492188
3: -6312.2260742, 17190.5605469, -6312.2260742, 17190.5605469, -23502.7871094, 23502.7851562
4: -20052.4492188, 11952.9658203, -20052.4492188, 11952.9658203, -32005.4121094, 32005.4140625

Time for backsubstitution: 0.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7271857, upper bound: 19342.8630890
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7231238, upper bound: 19342.7344048
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -15454.9707031, 11430.4335938, -17400.1269531, 13016.3066406, -28471.2773438, 28830.5605469
1: -12510.0507812, 11069.3183594, -14083.8212891, 12597.9062500, -25107.9570312, 25153.1406250
2: -18250.3164062, 12091.1337891, -20596.6406250, 13744.0898438, -31994.4062500, 32687.7734375
3: -6312.2260742, 17190.5605469, -7135.8437500, 19436.2500000, -25748.4765625, 24326.4023438
4: -20052.4492188, 11952.9658203, -22629.5566406, 13569.0703125, -33621.5195312, 34582.5234375

Time for backsubstitution: 0.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7271857, upper bound: 19342.8630890
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7231238, upper bound: 19342.7344048
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -17400.1269531, 13016.3066406, -15454.9707031, 11430.4335938, -28830.5605469, 28471.2753906
1: -14083.8212891, 12597.9062500, -12510.0507812, 11069.3183594, -25153.1406250, 25107.9570312
2: -20596.6406250, 13744.0898438, -18250.3164062, 12091.1337891, -32687.7734375, 31994.4062500
3: -7135.8437500, 19436.2500000, -6312.2260742, 17190.5605469, -24326.4042969, 25748.4746094
4: -22629.5566406, 13569.0703125, -20052.4492188, 11952.9658203, -34582.5234375, 33621.5195312

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8622535, upper bound: 19342.7271857
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7226587, upper bound: 19342.7226587
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -17400.1269531, 13016.3066406, -17400.1269531, 13016.3066406, -30416.4335938, 30416.4316406
1: -14083.8212891, 12597.9062500, -14083.8212891, 12597.9062500, -26681.7265625, 26681.7265625
2: -20596.6406250, 13744.0898438, -20596.6406250, 13744.0898438, -34340.7226562, 34340.7226562
3: -7135.8437500, 19436.2500000, -7135.8437500, 19436.2500000, -26572.0937500, 26572.0937500
4: -22629.5566406, 13569.0703125, -22629.5566406, 13569.0703125, -36198.6250000, 36198.6250000

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8622535, upper bound: 19342.7271857
time: 0.98 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7226587, upper bound: 19342.7226587
time: 0.88 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -15372.7880859, 11372.5957031, -16724.4414062, 12517.8974609, -27890.6855469, 28097.0351562
1: -12443.1738281, 11011.5703125, -13532.5087891, 12108.6621094, -24551.8359375, 24544.0781250
2: -18150.8046875, 12029.0634766, -19758.0351562, 13220.1113281, -31370.9160156, 31787.0976562
3: -6281.1396484, 17099.2949219, -6883.1884766, 18683.9394531, -24965.0781250, 23982.4843750
4: -19943.2949219, 11892.9472656, -21711.7167969, 13058.1513672, -33001.4453125, 33604.6640625

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8348020, upper bound: 19342.8739704
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8315735, upper bound: 19342.7535741
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -15454.9707031, 11430.4335938, -17638.8828125, 13212.0332031, -28667.0039062, 29069.3144531
1: -12510.0507812, 11069.3183594, -14275.6416016, 12788.5898438, -25298.6406250, 25344.9609375
2: -18250.3164062, 12091.1337891, -20875.1992188, 13953.5107422, -32203.8281250, 32966.3320312
3: -6312.2260742, 17190.5605469, -7240.7519531, 19711.6582031, -26023.8847656, 24431.3105469
4: -20052.4492188, 11952.9658203, -22937.2949219, 13768.6367188, -33821.0859375, 34890.2578125

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9464961, upper bound: 19342.9581449
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9464961, upper bound: 19342.9665842
time: 0.94 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -17400.1269531, 13016.3066406, -17336.7089844, 12989.7568359, -30389.8828125, 30353.0136719
1: -14083.8212891, 12597.9062500, -14032.3134766, 12576.0234375, -26659.8437500, 26630.2187500
2: -20596.6406250, 13744.0898438, -20523.5078125, 13721.3007812, -34317.9335938, 34267.5976562
3: -7135.8437500, 19436.2500000, -7118.8666992, 19383.9453125, -26519.7890625, 26555.1132812
4: -22629.5566406, 13569.0703125, -22548.2617188, 13537.5009766, -36167.0585938, 36117.3281250

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7461679, upper bound: 19342.8654797
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7369922, upper bound: 19342.7232734
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -17400.1269531, 13016.3066406, -19908.0585938, 15005.9785156, -32357.0781250, 32924.3632812
1: -14083.8212891, 12597.9062500, -16112.2021484, 14527.9570312, -28588.6777344, 28710.1093750
2: -20596.6406250, 13744.0898438, -23607.6992188, 15838.6308594, -36404.4804688, 37351.7890625
3: -7135.8437500, 19436.2500000, -8186.2548828, 22325.4218750, -29445.8554688, 27597.7324219
4: -22629.5566406, 13569.0703125, -25926.9453125, 15615.3916016, -38220.4062500, 39496.0156250

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7461679, upper bound: 19342.8654797
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7369922, upper bound: 19342.7232734
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -16724.4414062, 12517.8974609, -15372.7880859, 11372.5957031, -28097.0371094, 27890.6855469
1: -13532.5087891, 12108.6621094, -12443.1738281, 11011.5703125, -24544.0781250, 24551.8359375
2: -19758.0351562, 13220.1113281, -18150.8046875, 12029.0634766, -31787.0976562, 31370.9160156
3: -6883.1884766, 18683.9394531, -6281.1396484, 17099.2949219, -23982.4843750, 24965.0781250
4: -21711.7167969, 13058.1513672, -19943.2949219, 11892.9472656, -33604.6640625, 33001.4453125

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B1_A1_B1

### Relational analysis result of NS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8739704, upper bound: 19342.8348017
time: 0.80 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2

### Relational analysis result of NS_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7535741, upper bound: 19342.8315733
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -17638.8828125, 13212.0332031, -15454.9707031, 11430.4335938, -29069.3144531, 28667.0039062
1: -14275.6416016, 12788.5898438, -12510.0507812, 11069.3183594, -25344.9609375, 25298.6406250
2: -20875.1992188, 13953.5107422, -18250.3164062, 12091.1337891, -32966.3320312, 32203.8281250
3: -7240.7519531, 19711.6582031, -6312.2260742, 17190.5605469, -24431.3105469, 26023.8847656
4: -22937.2949219, 13768.6367188, -20052.4492188, 11952.9658203, -34890.2617188, 33821.0859375

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_B1_A2_A1

### Relational analysis result of NS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9581449, upper bound: 19342.9464961
time: 0.96 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2

### Relational analysis result of NS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9581449, upper bound: 19343.0392355
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -17336.7089844, 12989.7568359, -17336.7089844, 12989.7568359, -30326.4648438, 30326.4648438
1: -14032.3134766, 12576.0234375, -14032.3134766, 12576.0234375, -26608.3320312, 26608.3359375
2: -20523.5078125, 13721.3007812, -20523.5078125, 13721.3007812, -34244.8046875, 34244.8046875
3: -7118.8666992, 19383.9453125, -7118.8666992, 19383.9453125, -26502.8085938, 26502.8085938
4: -22548.2617188, 13537.5009766, -22548.2617188, 13537.5009766, -36085.7617188, 36085.7617188

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B2_A1_A1

### Relational analysis result of NS_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7504972, upper bound: 19342.8691429
time: 1.01 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2

### Relational analysis result of NS_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7477889, upper bound: 19342.7476948
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -19908.0585938, 15005.9785156, -17336.7089844, 12989.7568359, -32897.8164062, 32299.8632812
1: -16112.2021484, 14527.9570312, -14032.3134766, 12576.0234375, -28688.2226562, 28540.9023438
2: -23607.6992188, 15838.6308594, -20523.5078125, 13721.3007812, -37329.0000000, 36335.7656250
3: -8186.2548828, 22325.4218750, -7118.8666992, 19383.9453125, -27543.8203125, 29424.7109375
4: -25926.9453125, 15615.3916016, -22548.2617188, 13537.5009766, -39464.4453125, 38146.6484375

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8695830, upper bound: 19342.8380845
time: 2.52 seconds

## Relational analysis of NS_A2_B1_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7477889, upper bound: 19342.8354181
time: 1.04 seconds

## BFS NS instance: NS_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -17336.7089844, 12989.7568359, -17400.1269531, 13016.3066406, -30353.0156250, 30389.8828125
1: -14032.3134766, 12576.0234375, -14083.8212891, 12597.9062500, -26630.2187500, 26659.8437500
2: -20523.5078125, 13721.3007812, -20596.6406250, 13744.0898438, -34267.5937500, 34317.9335938
3: -7118.8666992, 19383.9453125, -7135.8437500, 19436.2500000, -26555.1152344, 26519.7890625
4: -22548.2617188, 13537.5009766, -22629.5566406, 13569.0703125, -36117.3281250, 36167.0585938

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_B1_A1_B1

### Relational analysis result of NS_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8654797, upper bound: 19342.7461679
time: 0.74 seconds

## Relational analysis of NS_A2_B2_B1_A1_B2

### Relational analysis result of NS_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7232734, upper bound: 19342.7369923
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -19908.0585938, 15005.9785156, -17400.1269531, 13016.3066406, -32924.3671875, 32357.0800781
1: -16112.2021484, 14527.9570312, -14083.8212891, 12597.9062500, -28710.1093750, 28588.6777344
2: -23607.6992188, 15838.6308594, -20596.6406250, 13744.0898438, -37351.7890625, 36404.4804688
3: -8186.2548828, 22325.4218750, -7135.8437500, 19436.2500000, -27597.7324219, 29445.8554688
4: -25926.9453125, 15615.3916016, -22629.5566406, 13569.0703125, -39496.0156250, 38220.4062500

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_B1_A2_B1

### Relational analysis result of NS_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8654797, upper bound: 19342.8324957
time: 0.87 seconds

## Relational analysis of NS_A2_B2_B1_A2_B2

### Relational analysis result of NS_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7232734, upper bound: 19342.8260442
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -17336.7089844, 12989.7568359, -19908.0585938, 15005.9785156, -32299.8632812, 32897.8164062
1: -14032.3134766, 12576.0234375, -16112.2021484, 14527.9570312, -28540.9023438, 28688.2226562
2: -20523.5078125, 13721.3007812, -23607.6992188, 15838.6308594, -36335.7656250, 37329.0000000
3: -7118.8666992, 19383.9453125, -8186.2548828, 22325.4218750, -29424.7109375, 27543.8203125
4: -22548.2617188, 13537.5009766, -25926.9453125, 15615.3916016, -38146.6484375, 39464.4453125

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_B2_A1_A1

### Relational analysis result of NS_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7504534, upper bound: 19342.8647031
time: 0.72 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2

### Relational analysis result of NS_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7477889, upper bound: 19342.7473947
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -19908.0585938, 15005.9785156, -19908.0585938, 15005.9785156, -34829.7500000, 34829.7500000
1: -16112.2021484, 14527.9570312, -16112.2021484, 14527.9570312, -30588.2285156, 30588.2285156
2: -23607.6992188, 15838.6308594, -23607.6992188, 15838.6308594, -39355.1679688, 39355.1679688
3: -8186.2548828, 22325.4218750, -8186.2548828, 22325.4218750, -30437.2910156, 30437.2910156
4: -25926.9453125, 15615.3916016, -25926.9453125, 15615.3916016, -41466.0078125, 41466.0078125

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_B2_A2_A1

### Relational analysis result of NS_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7504534, upper bound: 19342.9659420
time: 0.85 seconds

## Relational analysis of NS_A2_B2_B2_A2_A2

### Relational analysis result of NS_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7477889, upper bound: 19342.8470764
time: 0.75 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.45 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 3, lower bound: -19342.7271857, upper bound: 19342.8630890
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 3, lower bound: -19342.7231238, upper bound: 19342.7344048
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 3, lower bound: -19342.7271857, upper bound: 19342.8630890
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 3, lower bound: -19342.7231238, upper bound: 19342.7344048
NS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 3, lower bound: -19342.8622535, upper bound: 19342.7271857
NS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 3, lower bound: -19342.7226587, upper bound: 19342.7226587
NS_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 3, lower bound: -19342.8622535, upper bound: 19342.7271857
NS_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 3, lower bound: -19342.7226587, upper bound: 19342.7226587
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 3, lower bound: -19342.8348020, upper bound: 19342.8739704
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 3, lower bound: -19342.8315735, upper bound: 19342.7535741
NS_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 3, lower bound: -19342.9464961, upper bound: 19342.9581449
NS_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 3, lower bound: -19342.9464961, upper bound: 19342.9665842
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 3, lower bound: -19342.7461679, upper bound: 19342.8654797
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 3, lower bound: -19342.7369922, upper bound: 19342.7232734
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 3, lower bound: -19342.7461679, upper bound: 19342.8654797
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 3, lower bound: -19342.7369922, upper bound: 19342.7232734
NS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 3, lower bound: -19342.8739704, upper bound: 19342.8348017
NS_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 3, lower bound: -19342.7535741, upper bound: 19342.8315733
NS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 3, lower bound: -19342.9581449, upper bound: 19342.9464961
NS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 3, lower bound: -19342.9581449, upper bound: 19343.0392355
NS_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 3, lower bound: -19342.7504972, upper bound: 19342.8691429
NS_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 3, lower bound: -19342.7477889, upper bound: 19342.7476948
NS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 3, lower bound: -19342.8695830, upper bound: 19342.8380845
NS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 3, lower bound: -19342.7477889, upper bound: 19342.8354181
NS_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 3, lower bound: -19342.8654797, upper bound: 19342.7461679
NS_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 3, lower bound: -19342.7232734, upper bound: 19342.7369923
NS_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 3, lower bound: -19342.8654797, upper bound: 19342.8324957
NS_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 3, lower bound: -19342.7232734, upper bound: 19342.8260442
NS_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 3, lower bound: -19342.7504534, upper bound: 19342.8647031
NS_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 3, lower bound: -19342.7477889, upper bound: 19342.7473947
NS_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 3, lower bound: -19342.7504534, upper bound: 19342.9659420
NS_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 3, lower bound: -19342.7477889, upper bound: 19342.8470764

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -15079.4121094, 11150.6660156, -15221.3525391, 11256.1367188, -26335.5488281, 26372.0195312
1: -12206.4462891, 10800.2343750, -12321.1660156, 10901.6132812, -23108.0566406, 23121.4003906
2: -17805.7636719, 11797.1044922, -17973.8300781, 11907.5771484, -29713.3398438, 29770.9296875
3: -6159.3442383, 16770.0722656, -6216.8652344, 16929.0488281, -23088.3925781, 22986.9355469
4: -19564.1445312, 11660.5947266, -19748.7695312, 11770.7666016, -31334.9101562, 31409.3632812

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7431507, upper bound: 19342.7431507
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7431507, upper bound: 19342.7431507
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -17304.0332031, 12715.2548828, -15070.7919922, 11141.0771484, -28445.1093750, 27786.0468750
1: -14009.4804688, 12317.1445312, -12199.3447266, 10789.2568359, -24798.7382812, 24516.4863281
2: -20445.1972656, 13451.1572266, -17797.6210938, 11784.9736328, -32230.1699219, 31248.7773438
3: -7035.3032227, 19232.8066406, -6152.5722656, 16760.8710938, -23796.1738281, 25385.3769531
4: -22458.2949219, 13292.4462891, -19554.9160156, 11649.1259766, -34107.4218750, 32847.3593750

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7431507, upper bound: 19342.7431507
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7431507, upper bound: 19342.7431507
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -15079.4121094, 11150.6660156, -17169.8632812, 12845.1474609, -27924.5585938, 28320.5292969
1: -12206.4462891, 10800.2343750, -13897.4560547, 12433.2968750, -24639.7421875, 24697.6894531
2: -17805.7636719, 11797.1044922, -20323.9941406, 13564.2724609, -31370.0351562, 32121.0957031
3: -6159.3442383, 16770.0722656, -7041.8779297, 19178.5312500, -25337.8750000, 23811.9472656
4: -19564.1445312, 11660.5947266, -22330.0683594, 13390.8330078, -32954.9765625, 33990.6601562

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 24

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7231238, upper bound: 19342.7344048
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7231238, upper bound: 19342.7344048
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -17304.0332031, 12715.2548828, -17020.5000000, 12731.6865234, -30035.7187500, 29735.7519531
1: -14009.4804688, 12317.1445312, -13777.1279297, 12322.8974609, -26332.3789062, 26094.2734375
2: -20445.1972656, 13451.1572266, -20150.3339844, 13443.2832031, -33888.4804688, 33601.4921875
3: -7035.3032227, 19232.8066406, -6978.1533203, 19012.4472656, -26047.7500000, 26210.9589844
4: -22458.2949219, 13292.4462891, -22138.6601562, 13271.0869141, -35729.3750000, 35431.1054688

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7231238, upper bound: 19342.7344048
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7231238, upper bound: 19342.7344048
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -17169.8632812, 12845.1474609, -15079.4121094, 11150.6660156, -28320.5292969, 27924.5585938
1: -13897.4560547, 12433.2968750, -12206.4462891, 10800.2343750, -24697.6914062, 24639.7421875
2: -20323.9941406, 13564.2724609, -17805.7636719, 11797.1044922, -32121.0976562, 31370.0351562
3: -7041.8779297, 19178.5312500, -6159.3442383, 16770.0722656, -23811.9472656, 25337.8750000
4: -22330.0683594, 13390.8330078, -19564.1445312, 11660.5947266, -33990.6562500, 32954.9765625

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 24

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7344048, upper bound: 19342.7231238
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7344048, upper bound: 19342.7231238
time: 0.94 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -17020.5000000, 12731.6865234, -17304.0332031, 12715.2548828, -29735.7539062, 30035.7187500
1: -13777.1279297, 12322.8974609, -14009.4804688, 12317.1445312, -26094.2734375, 26332.3789062
2: -20150.3339844, 13443.2832031, -20445.1972656, 13451.1572266, -33601.4921875, 33888.4804688
3: -6978.1533203, 19012.4472656, -7035.3032227, 19232.8066406, -26210.9570312, 26047.7500000
4: -22138.6601562, 13271.0869141, -22458.2949219, 13292.4462891, -35431.1054688, 35729.3750000

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7344048, upper bound: 19342.7231238
time: 1.07 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7344048, upper bound: 19342.7231238
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -17169.8632812, 12845.1474609, -17028.9472656, 12740.8222656, -29910.6855469, 29874.0937500
1: -13897.4560547, 12433.2968750, -13783.3916016, 12333.3359375, -26230.7871094, 26216.6875000
2: -20323.9941406, 13564.2724609, -20156.9628906, 13454.8154297, -33778.8085938, 33721.2343750
3: -7041.8779297, 19178.5312500, -6984.7680664, 19020.7187500, -26062.5976562, 26163.2988281
4: -22330.0683594, 13390.8330078, -22146.6132812, 13282.3027344, -35612.3671875, 35537.4453125

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7226587, upper bound: 19342.7226587
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7226587, upper bound: 19342.7226587
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -17020.5000000, 12731.6865234, -19258.6230469, 14314.8486328, -31335.3457031, 31990.3085938
1: -13777.1279297, 12322.8974609, -15591.9550781, 13858.5380859, -27635.6640625, 27914.8515625
2: -20150.3339844, 13443.2832031, -22803.1542969, 15116.4033203, -35266.7382812, 36246.4375000
3: -6978.1533203, 19012.4472656, -7863.4697266, 21490.8242188, -28468.9765625, 26875.9179688
4: -22138.6601562, 13271.0869141, -25047.6210938, 14921.1982422, -37059.8554688, 38318.7031250

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7226587, upper bound: 19342.7226587
time: 1.19 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7226587, upper bound: 19342.7226587
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -14998.1894531, 11093.4384766, -16493.0585938, 12346.2333984, -27344.4218750, 27586.4960938
1: -12140.3447266, 10743.0820312, -13345.2978516, 11943.4794922, -24083.8183594, 24088.3789062
2: -17707.3925781, 11735.6181641, -19484.3144531, 13039.6767578, -30747.0703125, 31219.9316406
3: -6128.5795898, 16679.8183594, -6788.9838867, 18425.1250000, -24553.7031250, 23468.8007812
4: -19456.2363281, 11601.2041016, -21410.9980469, 12878.6132812, -32334.8476562, 33012.2031250

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8315735, upper bound: 19342.7535741
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8315735, upper bound: 19342.7535741
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -17222.2578125, 12657.5898438, -16356.8115234, 12242.9033203, -29465.1601562, 29014.4023438
1: -13942.8671875, 12259.5830078, -13235.2460938, 11842.4443359, -25785.3027344, 25494.8261719
2: -20346.2148438, 13389.2744141, -19325.5742188, 12929.4208984, -33275.6367188, 32714.8437500
3: -7004.4199219, 19141.8671875, -6731.1596680, 18273.6386719, -25278.0585938, 25873.0273438
4: -22349.6875000, 13232.6484375, -21236.6054688, 12769.4853516, -35119.1718750, 34469.2539062

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8315735, upper bound: 19342.7535741
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8315735, upper bound: 19342.7535741
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -15454.9707031, 11430.4335938, -17334.6777344, 12988.2441406, -28443.2148438, 28765.1113281
1: -12510.0507812, 11069.3183594, -14030.6728516, 12574.5673828, -25084.6171875, 25099.9902344
2: -18250.3164062, 12091.1337891, -20521.1503906, 13719.7246094, -31970.0410156, 32612.2851562
3: -6312.2260742, 17190.5605469, -7118.0292969, 19381.7148438, -25693.9375000, 24308.5839844
4: -20052.4492188, 11952.9658203, -22545.6621094, 13535.9228516, -33588.3710938, 34498.6289062

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7477827, upper bound: 19342.8677400
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7442406, upper bound: 19342.7435816
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -15454.9707031, 11430.4335938, -19888.4843750, 14991.1210938, -30415.0429688, 31318.9160156
1: -12510.0507812, 11069.3183594, -16096.2080078, 14513.2910156, -27013.7128906, 27165.5273438
2: -18250.3164062, 12091.1337891, -23583.5507812, 15822.9472656, -34073.2617188, 35674.6835938
3: -6312.2260742, 17190.5605469, -8178.8706055, 22303.1386719, -28615.3652344, 25369.4296875
4: -20052.4492188, 11952.9658203, -25900.5117188, 15600.1123047, -35652.5625000, 37853.4765625

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7477826, upper bound: 19342.8677400
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7442405, upper bound: 19342.7530863
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -17028.9472656, 12740.8222656, -17103.7636719, 12816.6484375, -29845.5898438, 29844.5859375
1: -13783.3916016, 12333.3359375, -13843.8349609, 12409.5009766, -26192.8925781, 26177.1699219
2: -20156.9628906, 13454.8154297, -20247.7949219, 13539.4638672, -33696.4257812, 33702.6093750
3: -6984.7680664, 19020.7187500, -7023.8383789, 19123.3027344, -26108.0703125, 26044.5566406
4: -22146.6132812, 13282.3027344, -22245.3574219, 13357.2636719, -35503.8750000, 35527.6562500

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7369922, upper bound: 19342.7232734
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7369922, upper bound: 19342.7232734
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -19258.6230469, 14314.8486328, -16965.5332031, 12711.3271484, -31969.9472656, 31280.3808594
1: -15591.9550781, 13858.5380859, -13732.4638672, 12306.4501953, -27898.4042969, 27591.0000000
2: -22803.1542969, 15116.4033203, -20087.0253906, 13426.7324219, -36229.8867188, 35203.4296875
3: -7863.4697266, 21490.8242188, -6964.4814453, 18969.4082031, -26832.8789062, 28455.3046875
4: -25047.6210938, 14921.1982422, -22068.1757812, 13245.8271484, -38293.4492188, 36989.3750000

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7369922, upper bound: 19342.7232734
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7369922, upper bound: 19342.7232734
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -17028.9472656, 12740.8222656, -19675.7714844, 14833.0869141, -31811.5488281, 32416.5937500
1: -13783.3916016, 12333.3359375, -15924.3339844, 14362.0781250, -28120.9531250, 28257.6660156
2: -20156.9628906, 13454.8154297, -23333.0234375, 15657.2148438, -35781.1875000, 36787.8398438
3: -6984.7680664, 19020.7187500, -8091.3261719, 22066.0566406, -29034.1679688, 27086.2167969
4: -22146.6132812, 13282.3027344, -25624.9921875, 15435.6269531, -37555.1718750, 38907.2890625

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7369923, upper bound: 19342.7232734
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7369923, upper bound: 19342.7232734
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -19258.6230469, 14314.8486328, -19528.8417969, 14720.9941406, -33896.2304688, 33843.6875000
1: -15591.9550781, 13858.5380859, -15806.2080078, 14253.1289062, -29794.4902344, 29664.7421875
2: -22803.1542969, 15116.4033203, -23162.7812500, 15537.8457031, -38274.3125000, 38279.1835938
3: -7863.4697266, 21490.8242188, -8028.7412109, 21903.6582031, -29737.6367188, 29482.7246094
4: -25047.6210938, 14921.1982422, -25437.4277344, 15317.3095703, -40301.1289062, 40358.6250000

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7369922, upper bound: 19342.7232734
time: 0.94 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7369922, upper bound: 19342.7232734
time: 1.08 seconds

## BFS NS instance: NS_A2_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -16493.0585938, 12346.2333984, -14998.1894531, 11093.4384766, -27586.4960938, 27344.4218750
1: -13345.2978516, 11943.4794922, -12140.3447266, 10743.0820312, -24088.3789062, 24083.8203125
2: -19484.3144531, 13039.6767578, -17707.3925781, 11735.6181641, -31219.9296875, 30747.0703125
3: -6788.9838867, 18425.1250000, -6128.5795898, 16679.8183594, -23468.8007812, 24553.7031250
4: -21410.9980469, 12878.6132812, -19456.2363281, 11601.2041016, -33012.2031250, 32334.8476562

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7535741, upper bound: 19342.8315731
time: 0.82 seconds

## Relational analysis of NS_A2_B1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7535741, upper bound: 19342.8315731
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -16356.8115234, 12242.9033203, -17222.2578125, 12657.5898438, -29014.4023438, 29465.1601562
1: -13235.2460938, 11842.4443359, -13942.8671875, 12259.5830078, -25494.8261719, 25785.3027344
2: -19325.5742188, 12929.4208984, -20346.2148438, 13389.2744141, -32714.8437500, 33275.6367188
3: -6731.1596680, 18273.6386719, -7004.4199219, 19141.8671875, -25873.0253906, 25278.0585938
4: -21236.6054688, 12769.4853516, -22349.6875000, 13232.6484375, -34469.2500000, 35119.1718750

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7535741, upper bound: 19342.8315731
time: 0.78 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7535741, upper bound: 19342.8315731
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -17334.6777344, 12988.2441406, -15454.9707031, 11430.4335938, -28765.1113281, 28443.2148438
1: -14030.6728516, 12574.5673828, -12510.0507812, 11069.3183594, -25099.9921875, 25084.6171875
2: -20521.1503906, 13719.7246094, -18250.3164062, 12091.1337891, -32612.2851562, 31970.0390625
3: -7118.0292969, 19381.7148438, -6312.2260742, 17190.5605469, -24308.5859375, 25693.9414062
4: -22545.6621094, 13535.9228516, -20052.4492188, 11952.9658203, -34498.6289062, 33588.3710938

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B1_A2_A1_B1

### Relational analysis result of NS_A2_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8677400, upper bound: 19342.7477826
time: 0.83 seconds

## Relational analysis of NS_A2_B1_B1_A2_A1_B2

### Relational analysis result of NS_A2_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7435816, upper bound: 19342.7442405
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -19888.4843750, 14991.1210938, -15454.9707031, 11430.4335938, -31318.9160156, 30415.0429688
1: -16096.2080078, 14513.2910156, -12510.0507812, 11069.3183594, -27165.5273438, 27013.7128906
2: -23583.5507812, 15822.9472656, -18250.3164062, 12091.1337891, -35674.6835938, 34073.2617188
3: -8178.8706055, 22303.1386719, -6312.2260742, 17190.5605469, -25369.4277344, 28615.3632812
4: -25900.5117188, 15600.1123047, -20052.4492188, 11952.9658203, -37853.4765625, 35652.5625000

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B1_A2_A2_B1

### Relational analysis result of NS_A2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8677400, upper bound: 19342.8268950
time: 1.11 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2_B2

### Relational analysis result of NS_A2_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7435816, upper bound: 19342.7442405
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -16960.5136719, 12710.3867188, -17103.7636719, 12816.6484375, -29777.1621094, 29814.1503906
1: -13727.9248047, 12307.4902344, -13843.8349609, 12409.5009766, -26137.4257812, 26151.3242188
2: -20078.0742188, 13428.0615234, -20247.7949219, 13539.4638672, -33617.5351562, 33675.8554688
3: -6965.6933594, 18962.8554688, -7023.8383789, 19123.3027344, -26088.9960938, 25986.6933594
4: -22058.8867188, 13246.7812500, -22245.3574219, 13357.2636719, -35416.1484375, 35492.1406250

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B2_A1_A1_B1

### Relational analysis result of NS_A2_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7477889, upper bound: 19342.7476948
time: 0.71 seconds

## Relational analysis of NS_A2_B1_B2_A1_A1_B2

### Relational analysis result of NS_A2_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7477889, upper bound: 19342.7476948
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -19204.0390625, 14295.5185547, -16965.5332031, 12711.3271484, -31915.3652344, 31261.0507812
1: -15547.9160156, 13842.3193359, -13732.4638672, 12306.4501953, -27854.3613281, 27574.7812500
2: -22741.0097656, 15099.6142578, -20087.0253906, 13426.7324219, -36167.7421875, 35186.6406250
3: -7851.1518555, 21447.9667969, -6964.4814453, 18969.4082031, -26820.5605469, 28412.4453125
4: -24978.7636719, 14896.4580078, -22068.1757812, 13245.8271484, -38224.5859375, 36964.6328125

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B2_A1_A2_B1

### Relational analysis result of NS_A2_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7477889, upper bound: 19342.7476948
time: 0.80 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2_B2

### Relational analysis result of NS_A2_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7477889, upper bound: 19342.7476948
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -19675.7714844, 14833.0869141, -16960.5136719, 12710.3867188, -32386.1582031, 31749.2578125
1: -15924.3339844, 14362.0781250, -13727.9248047, 12307.4902344, -28231.8242188, 28069.1718750
2: -23333.0234375, 15657.2148438, -20078.0742188, 13428.0615234, -36761.0859375, 35706.6250000
3: -8091.3261719, 22066.0566406, -6965.6933594, 18962.8554688, -27026.7011719, 29010.9433594
4: -25624.9921875, 15435.6269531, -22058.8867188, 13246.7812500, -38871.7734375, 37474.8906250

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7579544, upper bound: 19342.8354187
time: 0.80 seconds

## Relational analysis of NS_A2_B1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7579544, upper bound: 19342.8354181
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -19528.8417969, 14720.9941406, -19204.0390625, 14295.5185547, -33824.3593750, 33847.5195312
1: -15806.2080078, 14253.1289062, -15547.9160156, 13842.3193359, -29648.5234375, 29753.8339844
2: -23162.7812500, 15537.8457031, -22741.0097656, 15099.6142578, -38262.3945312, 38216.1640625
3: -8028.7412109, 21903.6582031, -7851.1518555, 21447.9667969, -29438.5625000, 29719.3105469
4: -25437.4277344, 15317.3095703, -24978.7636719, 14896.4580078, -40333.8867188, 40239.3945312

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7579544, upper bound: 19342.8354179
time: 0.76 seconds

## Relational analysis of NS_A2_B1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7579544, upper bound: 19342.8354187
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -17103.7636719, 12816.6484375, -17028.9472656, 12740.8222656, -29844.5859375, 29845.5898438
1: -13843.8349609, 12409.5009766, -13783.3916016, 12333.3359375, -26177.1699219, 26192.8925781
2: -20247.7949219, 13539.4638672, -20156.9628906, 13454.8154297, -33702.6093750, 33696.4257812
3: -7023.8383789, 19123.3027344, -6984.7680664, 19020.7187500, -26044.5566406, 26108.0703125
4: -22245.3574219, 13357.2636719, -22146.6132812, 13282.3027344, -35527.6562500, 35503.8750000

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7232734, upper bound: 19342.7369922
time: 1.61 seconds

## Relational analysis of NS_A2_B2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7232734, upper bound: 19342.7369922
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -16965.5332031, 12711.3271484, -19258.6230469, 14314.8486328, -31280.3808594, 31969.9492188
1: -13732.4638672, 12306.4501953, -15591.9550781, 13858.5380859, -27591.0000000, 27898.4042969
2: -20087.0253906, 13426.7324219, -22803.1542969, 15116.4033203, -35203.4296875, 36229.8867188
3: -6964.4814453, 18969.4082031, -7863.4697266, 21490.8242188, -28455.3046875, 26832.8789062
4: -22068.1757812, 13245.8271484, -25047.6210938, 14921.1982422, -36989.3750000, 38293.4492188

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7232734, upper bound: 19342.7369923
time: 0.88 seconds

## Relational analysis of NS_A2_B2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7232734, upper bound: 19342.7369923
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -19675.7714844, 14833.0869141, -17028.9472656, 12740.8222656, -32416.5937500, 31811.5507812
1: -15924.3339844, 14362.0781250, -13783.3916016, 12333.3359375, -28257.6660156, 28120.9531250
2: -23333.0234375, 15657.2148438, -20156.9628906, 13454.8154297, -36787.8398438, 35781.1875000
3: -8091.3261719, 22066.0566406, -6984.7680664, 19020.7187500, -27086.2167969, 29034.1679688
4: -25624.9921875, 15435.6269531, -22146.6132812, 13282.3027344, -38907.2929688, 37555.1718750

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7367790, upper bound: 19342.8260443
time: 0.92 seconds

## Relational analysis of NS_A2_B2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7367790, upper bound: 19342.8260428
time: 9.16 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -19528.8417969, 14720.9941406, -19258.6230469, 14314.8486328, -33843.6914062, 33896.2343750
1: -15806.2080078, 14253.1289062, -15591.9550781, 13858.5380859, -29664.7421875, 29794.4902344
2: -23162.7812500, 15537.8457031, -22803.1542969, 15116.4033203, -38279.1835938, 38274.3125000
3: -8028.7412109, 21903.6582031, -7863.4697266, 21490.8242188, -29482.7246094, 29737.6386719
4: -25437.4277344, 15317.3095703, -25047.6210938, 14921.1982422, -40358.6250000, 40301.1289062

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7367790, upper bound: 19342.8260442
time: 0.85 seconds

## Relational analysis of NS_A2_B2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7367790, upper bound: 19342.8260445
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -16960.5136719, 12710.3867188, -19675.7714844, 14833.0869141, -31749.2578125, 32386.1582031
1: -13727.9248047, 12307.4902344, -15924.3339844, 14362.0781250, -28069.1699219, 28231.8242188
2: -20078.0742188, 13428.0615234, -23333.0234375, 15657.2148438, -35706.6250000, 36761.0859375
3: -6965.6933594, 18962.8554688, -8091.3261719, 22066.0566406, -29010.9394531, 27026.7031250
4: -22058.8867188, 13246.7812500, -25624.9921875, 15435.6269531, -37474.8945312, 38871.7734375

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_B2_A1_A1_B1

### Relational analysis result of NS_A2_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7492198, upper bound: 19342.7473947
time: 0.77 seconds

## Relational analysis of NS_A2_B2_B2_A1_A1_B2

### Relational analysis result of NS_A2_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7492198, upper bound: 19342.7473947
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -19204.0390625, 14295.5185547, -19528.8417969, 14720.9941406, -33847.5195312, 33824.3593750
1: -15547.9160156, 13842.3193359, -15806.2080078, 14253.1289062, -29753.8339844, 29648.5234375
2: -22741.0097656, 15099.6142578, -23162.7812500, 15537.8457031, -38216.1601562, 38262.3945312
3: -7851.1518555, 21447.9667969, -8028.7412109, 21903.6582031, -29719.3105469, 29438.5625000
4: -24978.7636719, 14896.4580078, -25437.4277344, 15317.3095703, -40239.3984375, 40333.8867188

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_B2_A1_A2_B1

### Relational analysis result of NS_A2_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7492198, upper bound: 19342.7473947
time: 0.80 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2_B2

### Relational analysis result of NS_A2_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7492198, upper bound: 19342.7473947
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -19533.1210938, 14727.0341797, -19675.7714844, 14833.0869141, -34279.6914062, 34315.5039062
1: -15808.9785156, 14260.6083984, -15924.3339844, 14362.0781250, -30117.1210938, 30130.0761719
2: -23164.1621094, 15546.1416016, -23333.0234375, 15657.2148438, -38727.0000000, 38783.6796875
3: -8033.3149414, 21906.6015625, -8091.3261719, 22066.0566406, -30022.8203125, 29921.7363281
4: -25439.3359375, 15325.4531250, -25624.9921875, 15435.6269531, -40794.9492188, 40869.2226562

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_B2_A2_A1_B1

### Relational analysis result of NS_A2_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8469320, upper bound: 19342.8470761
time: 0.79 seconds

## Relational analysis of NS_A2_B2_B2_A2_A1_B2

### Relational analysis result of NS_A2_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8469320, upper bound: 19342.8470761
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -21769.1367188, 16309.0244141, -19528.8417969, 14720.9941406, -36373.7500000, 35737.0898438
1: -17622.4628906, 15791.5576172, -15806.2080078, 14253.1289062, -31797.7929688, 31534.8300781
2: -25817.1386719, 17213.8867188, -23162.7812500, 15537.8457031, -41229.8632812, 40267.7031250
3: -8916.7792969, 24385.1230469, -8028.7412109, 21903.6582031, -30729.5136719, 32327.1777344
4: -28349.9394531, 16971.6445312, -25437.4277344, 15317.3095703, -43553.3632812, 42312.5078125

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_B2_A2_A2_B1

### Relational analysis result of NS_A2_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8469320, upper bound: 19342.8470761
time: 0.87 seconds

## Relational analysis of NS_A2_B2_B2_A2_A2_B2

### Relational analysis result of NS_A2_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8469320, upper bound: 19342.8470762
time: 0.76 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.36 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7431507, upper bound: 19342.7431507
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7431507, upper bound: 19342.7431507
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7431507, upper bound: 19342.7431507
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7431507, upper bound: 19342.7431507
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7231238, upper bound: 19342.7344048
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7231238, upper bound: 19342.7344048
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7231238, upper bound: 19342.7344048
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7231238, upper bound: 19342.7344048
NS_A1_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7344048, upper bound: 19342.7231238
NS_A1_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7344048, upper bound: 19342.7231238
NS_A1_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7344048, upper bound: 19342.7231238
NS_A1_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7344048, upper bound: 19342.7231238
NS_A1_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7226587, upper bound: 19342.7226587
NS_A1_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7226587, upper bound: 19342.7226587
NS_A1_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7226587, upper bound: 19342.7226587
NS_A1_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7226587, upper bound: 19342.7226587
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.8315735, upper bound: 19342.7535741
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.8315735, upper bound: 19342.7535741
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.8315735, upper bound: 19342.7535741
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.8315735, upper bound: 19342.7535741
NS_A1_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7477827, upper bound: 19342.8677400
NS_A1_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7442406, upper bound: 19342.7435816
NS_A1_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7477826, upper bound: 19342.8677400
NS_A1_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7442405, upper bound: 19342.7530863
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7369922, upper bound: 19342.7232734
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7369922, upper bound: 19342.7232734
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7369922, upper bound: 19342.7232734
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7369922, upper bound: 19342.7232734
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7369923, upper bound: 19342.7232734
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7369923, upper bound: 19342.7232734
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7369922, upper bound: 19342.7232734
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7369922, upper bound: 19342.7232734
NS_A2_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7535741, upper bound: 19342.8315731
NS_A2_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7535741, upper bound: 19342.8315731
NS_A2_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7535741, upper bound: 19342.8315731
NS_A2_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7535741, upper bound: 19342.8315731
NS_A2_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.8677400, upper bound: 19342.7477826
NS_A2_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7435816, upper bound: 19342.7442405
NS_A2_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.8677400, upper bound: 19342.8268950
NS_A2_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7435816, upper bound: 19342.7442405
NS_A2_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7477889, upper bound: 19342.7476948
NS_A2_B1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7477889, upper bound: 19342.7476948
NS_A2_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7477889, upper bound: 19342.7476948
NS_A2_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7477889, upper bound: 19342.7476948
NS_A2_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7579544, upper bound: 19342.8354187
NS_A2_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7579544, upper bound: 19342.8354181
NS_A2_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7579544, upper bound: 19342.8354179
NS_A2_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7579544, upper bound: 19342.8354187
NS_A2_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7232734, upper bound: 19342.7369922
NS_A2_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7232734, upper bound: 19342.7369922
NS_A2_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7232734, upper bound: 19342.7369923
NS_A2_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7232734, upper bound: 19342.7369923
NS_A2_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7367790, upper bound: 19342.8260443
NS_A2_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7367790, upper bound: 19342.8260428
NS_A2_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7367790, upper bound: 19342.8260442
NS_A2_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7367790, upper bound: 19342.8260445
NS_A2_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7492198, upper bound: 19342.7473947
NS_A2_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7492198, upper bound: 19342.7473947
NS_A2_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7492198, upper bound: 19342.7473947
NS_A2_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.7492198, upper bound: 19342.7473947
NS_A2_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.8469320, upper bound: 19342.8470761
NS_A2_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.8469320, upper bound: 19342.8470761
NS_A2_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.8469320, upper bound: 19342.8470761
NS_A2_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 3, lower bound: -19342.8469320, upper bound: 19342.8470762

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -15079.4121094, 11150.6660156, -15079.4121094, 11150.6660156, -26230.0781250, 26230.0781250
1: -12206.4462891, 10800.2343750, -12206.4462891, 10800.2343750, -23006.6796875, 23006.6796875
2: -17805.7636719, 11797.1044922, -17805.7636719, 11797.1044922, -29602.8671875, 29602.8671875
3: -6159.3442383, 16770.0722656, -6159.3442383, 16770.0722656, -22929.4140625, 22929.4140625
4: -19564.1445312, 11660.5947266, -19564.1445312, 11660.5947266, -31224.7382812, 31224.7382812

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7157609, upper bound: 19342.8435579
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7109404, upper bound: 19342.8435579
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -15079.4121094, 11150.6660156, -17304.0332031, 12715.2548828, -27794.6679688, 28454.6992188
1: -12206.4462891, 10800.2343750, -14009.4804688, 12317.1445312, -24523.5898438, 24809.7148438
2: -17805.7636719, 11797.1044922, -20445.1972656, 13451.1572266, -31256.9218750, 32242.3007812
3: -6159.3442383, 16770.0722656, -7035.3032227, 19232.8066406, -25392.1503906, 23805.3750000
4: -19564.1445312, 11660.5947266, -22458.2949219, 13292.4462891, -32856.5898438, 34118.8906250

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19340.5577195, upper bound: 19342.8392300
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19340.4319644, upper bound: 19340.5482198
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -17304.0332031, 12715.2548828, -15079.4121094, 11150.6660156, -28454.6992188, 27794.6679688
1: -14009.4804688, 12317.1445312, -12206.4462891, 10800.2343750, -24809.7148438, 24523.5898438
2: -20445.1972656, 13451.1572266, -17805.7636719, 11797.1044922, -32242.3007812, 31256.9218750
3: -7035.3032227, 19232.8066406, -6159.3442383, 16770.0722656, -23805.3750000, 25392.1503906
4: -22458.2949219, 13292.4462891, -19564.1445312, 11660.5947266, -34118.8867188, 32856.5898438

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7397225, upper bound: 19342.6776208
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19340.5482198, upper bound: 19340.9681810
time: 1.00 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -17304.0332031, 12715.2548828, -17304.0332031, 12715.2548828, -30019.2890625, 30019.2890625
1: -14009.4804688, 12317.1445312, -14009.4804688, 12317.1445312, -26326.6250000, 26326.6250000
2: -20445.1972656, 13451.1572266, -20445.1972656, 13451.1572266, -33896.3554688, 33896.3554688
3: -7035.3032227, 19232.8066406, -7035.3032227, 19232.8066406, -26268.1093750, 26268.1093750
4: -22458.2949219, 13292.4462891, -22458.2949219, 13292.4462891, -35750.7421875, 35750.7421875

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7134936, upper bound: 19342.7190485
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7085000, upper bound: 19342.7085000
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -15079.4121094, 11150.6660156, -17028.9472656, 12740.8222656, -27820.2343750, 28179.6113281
1: -12206.4462891, 10800.2343750, -13783.3916016, 12333.3359375, -24539.7773438, 24583.6250000
2: -17805.7636719, 11797.1044922, -20156.9628906, 13454.8154297, -31260.5781250, 31954.0664062
3: -6159.3442383, 16770.0722656, -6984.7680664, 19020.7187500, -25180.0625000, 23754.8359375
4: -19564.1445312, 11660.5947266, -22146.6132812, 13282.3027344, -32846.4414062, 33807.2070312

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7181726, upper bound: 19342.8600947
time: 0.97 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7107537, upper bound: 19342.8436747
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -15079.4121094, 11150.6660156, -19258.6230469, 14314.8486328, -29394.2617188, 30409.2890625
1: -12206.4462891, 10800.2343750, -15591.9550781, 13858.5380859, -26064.9804688, 26392.1894531
2: -17805.7636719, 11797.1044922, -22803.1542969, 15116.4033203, -32922.1640625, 34600.2578125
3: -6159.3442383, 16770.0722656, -7863.4697266, 21490.8242188, -27650.1679688, 24633.5410156
4: -19564.1445312, 11660.5947266, -25047.6210938, 14921.1982422, -34485.3437500, 36708.2109375

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7153835, upper bound: 19342.8103019
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7168218, upper bound: 19342.8026693
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -17304.0332031, 12715.2548828, -17028.9472656, 12740.8222656, -30044.8554688, 29744.1992188
1: -14009.4804688, 12317.1445312, -13783.3916016, 12333.3359375, -26342.8125000, 26100.5351562
2: -20445.1972656, 13451.1572266, -20156.9628906, 13454.8154297, -33900.0117188, 33608.1210938
3: -7035.3032227, 19232.8066406, -6984.7680664, 19020.7187500, -26056.0214844, 26217.5742188
4: -22458.2949219, 13292.4462891, -22146.6132812, 13282.3027344, -35740.5976562, 35439.0585938

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7135128, upper bound: 19342.7125960
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7081476, upper bound: 19342.7088192
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -17304.0332031, 12715.2548828, -19258.6230469, 14314.8486328, -31618.8828125, 31973.8769531
1: -14009.4804688, 12317.1445312, -15591.9550781, 13858.5380859, -27868.0156250, 27909.0996094
2: -20445.1972656, 13451.1572266, -22803.1542969, 15116.4033203, -35561.6015625, 36254.3125000
3: -7035.3032227, 19232.8066406, -7863.4697266, 21490.8242188, -28526.1269531, 27096.2773438
4: -22458.2949219, 13292.4462891, -25047.6210938, 14921.1982422, -37379.4921875, 38340.0664062

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7135128, upper bound: 19342.7125960
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7081476, upper bound: 19342.7088192
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -17028.9472656, 12740.8222656, -15079.4121094, 11150.6660156, -28179.6113281, 27820.2343750
1: -13783.3916016, 12333.3359375, -12206.4462891, 10800.2343750, -24583.6250000, 24539.7773438
2: -20156.9628906, 13454.8154297, -17805.7636719, 11797.1044922, -31954.0664062, 31260.5761719
3: -6984.7680664, 19020.7187500, -6159.3442383, 16770.0722656, -23754.8378906, 25180.0625000
4: -22146.6132812, 13282.3027344, -19564.1445312, 11660.5947266, -33807.2070312, 32846.4453125

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8600947, upper bound: 19342.7181726
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_A1_A2

### Relational analysis result of NS_A1_B1_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8436747, upper bound: 19342.7107537
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -19258.6230469, 14314.8486328, -15079.4121094, 11150.6660156, -30409.2890625, 29394.2597656
1: -15591.9550781, 13858.5380859, -12206.4462891, 10800.2343750, -26392.1894531, 26064.9804688
2: -22803.1542969, 15116.4033203, -17805.7636719, 11797.1044922, -34600.2578125, 32922.1640625
3: -7863.4697266, 21490.8242188, -6159.3442383, 16770.0722656, -24633.5410156, 27650.1679688
4: -25047.6210938, 14921.1982422, -19564.1445312, 11660.5947266, -36708.2109375, 34485.3437500

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8103019, upper bound: 19342.7153835
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8026693, upper bound: 19342.7168218
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -17028.9472656, 12740.8222656, -17304.0332031, 12715.2548828, -29744.1992188, 30044.8554688
1: -13783.3916016, 12333.3359375, -14009.4804688, 12317.1445312, -26100.5351562, 26342.8125000
2: -20156.9628906, 13454.8154297, -20445.1972656, 13451.1572266, -33608.1210938, 33900.0117188
3: -6984.7680664, 19020.7187500, -7035.3032227, 19232.8066406, -26217.5742188, 26056.0214844
4: -22146.6132812, 13282.3027344, -22458.2949219, 13292.4462891, -35439.0585938, 35740.5976562

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_B2_A1_A1

### Relational analysis result of NS_A1_B1_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7125960, upper bound: 19342.7135128
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_A1_A2

### Relational analysis result of NS_A1_B1_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7088192, upper bound: 19342.7081476
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -19258.6230469, 14314.8486328, -17304.0332031, 12715.2548828, -31973.8750000, 31618.8828125
1: -15591.9550781, 13858.5380859, -14009.4804688, 12317.1445312, -27909.0996094, 27868.0156250
2: -22803.1542969, 15116.4033203, -20445.1972656, 13451.1572266, -36254.3125000, 35561.6015625
3: -7863.4697266, 21490.8242188, -7035.3032227, 19232.8066406, -27096.2773438, 28526.1269531
4: -25047.6210938, 14921.1982422, -22458.2949219, 13292.4462891, -38340.0664062, 37379.4921875

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_B2_A2_A1

### Relational analysis result of NS_A1_B1_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7125960, upper bound: 19342.7135128
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_A2_A2

### Relational analysis result of NS_A1_B1_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7088192, upper bound: 19342.7081476
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -17028.9472656, 12740.8222656, -17028.9472656, 12740.8222656, -29769.7695312, 29769.7695312
1: -13783.3916016, 12333.3359375, -13783.3916016, 12333.3359375, -26116.7265625, 26116.7265625
2: -20156.9628906, 13454.8154297, -20156.9628906, 13454.8154297, -33611.7773438, 33611.7773438
3: -6984.7680664, 19020.7187500, -6984.7680664, 19020.7187500, -26005.4863281, 26005.4863281
4: -22146.6132812, 13282.3027344, -22146.6132812, 13282.3027344, -35428.9140625, 35428.9140625

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B2_B1_A1_A1

### Relational analysis result of NS_A1_B1_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8580181, upper bound: 19342.7181726
time: 1.12 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_A1_A2

### Relational analysis result of NS_A1_B1_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8428644, upper bound: 19342.7107537
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -19258.6230469, 14314.8486328, -17028.9472656, 12740.8222656, -31999.4453125, 31343.7910156
1: -15591.9550781, 13858.5380859, -13783.3916016, 12333.3359375, -27925.2890625, 27641.9296875
2: -22803.1542969, 15116.4033203, -20156.9628906, 13454.8154297, -36257.9687500, 35273.3671875
3: -7863.4697266, 21490.8242188, -6984.7680664, 19020.7187500, -26884.1875000, 28475.5917969
4: -25047.6210938, 14921.1982422, -22146.6132812, 13282.3027344, -38329.9218750, 37067.8125000

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8428761, upper bound: 19342.7157363
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8428644, upper bound: 19342.7107537
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -17028.9472656, 12740.8222656, -19258.6230469, 14314.8486328, -31343.7929688, 31999.4453125
1: -13783.3916016, 12333.3359375, -15591.9550781, 13858.5380859, -27641.9296875, 27925.2871094
2: -20156.9628906, 13454.8154297, -22803.1542969, 15116.4033203, -35273.3671875, 36257.9687500
3: -6984.7680664, 19020.7187500, -7863.4697266, 21490.8242188, -28475.5917969, 26884.1875000
4: -22146.6132812, 13282.3027344, -25047.6210938, 14921.1982422, -37067.8125000, 38329.9218750

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_A1

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7125960, upper bound: 19342.7135128
time: 1.08 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_A2

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7083976, upper bound: 19342.7081476
time: 1.01 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -19258.6230469, 14314.8486328, -19258.6230469, 14314.8486328, -33573.4726562, 33573.4687500
1: -15591.9550781, 13858.5380859, -15591.9550781, 13858.5380859, -29450.4921875, 29450.4921875
2: -22803.1542969, 15116.4033203, -22803.1542969, 15116.4033203, -37919.5585938, 37919.5585938
3: -7863.4697266, 21490.8242188, -7863.4697266, 21490.8242188, -29354.2929688, 29354.2929688
4: -25047.6210938, 14921.1982422, -25047.6210938, 14921.1982422, -39968.8203125, 39968.8203125

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B2_B2_A2_A1

### Relational analysis result of NS_A1_B1_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7125960, upper bound: 19342.7135128
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_A2_A2

### Relational analysis result of NS_A1_B1_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7083976, upper bound: 19342.7081476
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -14998.1894531, 11093.4384766, -16351.5117188, 12241.2968750, -27239.4863281, 27444.9492188
1: -12140.3447266, 10743.0820312, -13230.7705078, 11842.6308594, -23982.9707031, 23973.8515625
2: -17707.3925781, 11735.6181641, -19316.7578125, 12929.6152344, -30637.0078125, 31052.3730469
3: -6128.5795898, 16679.8183594, -6731.5532227, 18266.6347656, -24395.2148438, 23411.3710938
4: -19456.2363281, 11601.2041016, -21226.9355469, 12769.4882812, -32225.7246094, 32828.1406250

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8159368, upper bound: 19342.8695451
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8003056, upper bound: 19342.8546098
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -14998.1894531, 11093.4384766, -18572.7890625, 13806.0390625, -28804.2285156, 29666.2265625
1: -12140.3447266, 10743.0820312, -15031.4326172, 13358.7333984, -25499.0742188, 25774.5097656
2: -17707.3925781, 11735.6181641, -21952.7890625, 14581.8896484, -32289.2812500, 33688.4062500
3: -6128.5795898, 16679.8183594, -7607.4916992, 20723.9589844, -26852.5351562, 24287.3105469
4: -19456.2363281, 11601.2041016, -24118.3652344, 14400.0126953, -33856.2421875, 35719.5664062

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8181931, upper bound: 19342.8213629
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8195324, upper bound: 19342.8146781
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -17222.2578125, 12657.5898438, -16351.5117188, 12241.2968750, -29463.5546875, 29009.1015625
1: -13942.8671875, 12259.5830078, -13230.7705078, 11842.6308594, -25785.4941406, 25490.3515625
2: -20346.2148438, 13389.2744141, -19316.7578125, 12929.6152344, -33275.8281250, 32706.0273438
3: -7004.4199219, 19141.8671875, -6731.5532227, 18266.6347656, -25271.0546875, 25873.4199219
4: -22349.6875000, 13232.6484375, -21226.9355469, 12769.4882812, -35119.1757812, 34459.5820312

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8111708, upper bound: 19342.7270787
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7980017, upper bound: 19342.7233665
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -17222.2578125, 12657.5898438, -18572.7890625, 13806.0390625, -31028.2949219, 31230.3789062
1: -13942.8671875, 12259.5830078, -15031.4326172, 13358.7333984, -27301.5976562, 27291.0117188
2: -20346.2148438, 13389.2744141, -21952.7890625, 14581.8896484, -34928.1054688, 35342.0625000
3: -7004.4199219, 19141.8671875, -7607.4916992, 20723.9589844, -27728.3789062, 26749.3593750
4: -22349.6875000, 13232.6484375, -24118.3652344, 14400.0126953, -36749.6992188, 37351.0156250

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8111712, upper bound: 19342.7270787
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7980017, upper bound: 19342.7233665
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -15079.4121094, 11150.6660156, -17101.7656250, 12815.1484375, -27894.5605469, 28252.4316406
1: -12206.4462891, 10800.2343750, -13842.2177734, 12408.0615234, -24614.5039062, 24642.4531250
2: -17805.7636719, 11797.1044922, -20245.4765625, 13537.8974609, -31343.6601562, 32042.5800781
3: -6159.3442383, 16770.0722656, -7023.0063477, 19121.0957031, -25280.4394531, 23793.0761719
4: -19564.1445312, 11660.5947266, -22242.7949219, 13355.7001953, -32919.8437500, 33903.3906250

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7442405, upper bound: 19342.7435816
time: 1.00 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7442405, upper bound: 19342.7435816
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -17304.0332031, 12715.2548828, -16963.5136719, 12709.8134766, -30013.8476562, 29678.7656250
1: -14009.4804688, 12317.1445312, -13730.8281250, 12305.0019531, -26314.4824219, 26047.9726562
2: -20445.1972656, 13451.1572266, -20084.6816406, 13425.1523438, -33870.3515625, 33535.8398438
3: -7035.3032227, 19232.8066406, -6963.6406250, 18967.1914062, -26002.4941406, 26196.4472656
4: -22458.2949219, 13292.4462891, -22065.5898438, 13244.2460938, -35702.5390625, 35358.0351562

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7442405, upper bound: 19342.7435816
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7442405, upper bound: 19342.7435816
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -15079.4121094, 11150.6660156, -19656.3281250, 14818.3369141, -29865.4375000, 30806.9941406
1: -12206.4462891, 10800.2343750, -15908.4462891, 14347.5273438, -26542.9453125, 26708.6796875
2: -17805.7636719, 11797.1044922, -23309.0292969, 15641.6455078, -33447.4101562, 35106.1250000
3: -6159.3442383, 16770.0722656, -8083.9985352, 22043.9179688, -28203.2617188, 24854.0703125
4: -19564.1445312, 11660.5947266, -25598.7207031, 15420.4580078, -34984.6015625, 37259.3164062

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8236629, upper bound: 19342.7530863
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8236629, upper bound: 19342.7530863
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -17304.0332031, 12715.2548828, -19510.4570312, 14707.0664062, -31942.7382812, 32225.7109375
1: -14009.4804688, 12317.1445312, -15791.1826172, 14239.3886719, -28209.3398438, 28108.3261719
2: -20445.1972656, 13451.1572266, -23140.1054688, 15523.1376953, -35933.2578125, 36591.2578125
3: -7035.3032227, 19232.8066406, -8021.8105469, 21882.7363281, -28918.0390625, 27246.5527344
4: -22458.2949219, 13292.4462891, -25412.6054688, 15302.9814453, -37724.6210938, 38705.0507812

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8236626, upper bound: 19342.7530863
time: 1.08 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8236626, upper bound: 19342.7530863
time: 0.97 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -17028.9472656, 12740.8222656, -16960.5136719, 12710.3867188, -29739.3300781, 29701.3359375
1: -13783.3916016, 12333.3359375, -13727.9248047, 12307.4902344, -26090.8828125, 26061.2558594
2: -20156.9628906, 13454.8154297, -20078.0742188, 13428.0615234, -33585.0234375, 33532.8867188
3: -6984.7680664, 19020.7187500, -6965.6933594, 18962.8554688, -25947.6230469, 25986.4121094
4: -22146.6132812, 13282.3027344, -22058.8867188, 13246.7812500, -35393.3945312, 35341.1875000

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7266415, upper bound: 19342.8583178
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7120750, upper bound: 19342.8441606
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -17028.9472656, 12740.8222656, -19204.0390625, 14295.5185547, -31324.4648438, 31944.8613281
1: -13783.3916016, 12333.3359375, -15547.9160156, 13842.3193359, -27625.7109375, 27881.2480469
2: -20156.9628906, 13454.8154297, -22741.0097656, 15099.6142578, -35256.5781250, 36195.8242188
3: -6984.7680664, 19020.7187500, -7851.1518555, 21447.9667969, -28432.7324219, 26871.8710938
4: -22146.6132812, 13282.3027344, -24978.7636719, 14896.4580078, -37043.0703125, 38261.0585938

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7159401, upper bound: 19342.8441606
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7120750, upper bound: 19342.8441606
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -19258.6230469, 14314.8486328, -16960.5136719, 12710.3867188, -31969.0097656, 31275.3632812
1: -15591.9550781, 13858.5380859, -13727.9248047, 12307.4902344, -27899.4453125, 27586.4609375
2: -22803.1542969, 15116.4033203, -20078.0742188, 13428.0615234, -36231.2148438, 35194.4765625
3: -7863.4697266, 21490.8242188, -6965.6933594, 18962.8554688, -26826.3242188, 28456.5175781
4: -25047.6210938, 14921.1982422, -22058.8867188, 13246.7812500, -38294.4023438, 36980.0859375

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7322408, upper bound: 19342.6588622
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7191075, upper bound: 19342.7134211
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7091618, upper bound: 19342.7082326
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -19258.6230469, 14314.8486328, -19204.0390625, 14295.5185547, -33554.1406250, 33518.8867188
1: -15591.9550781, 13858.5380859, -15547.9160156, 13842.3193359, -29434.2734375, 29406.4511719
2: -22803.1542969, 15116.4033203, -22741.0097656, 15099.6142578, -37902.7656250, 37857.4140625
3: -7863.4697266, 21490.8242188, -7851.1518555, 21447.9667969, -29311.4335938, 29341.9726562
4: -25047.6210938, 14921.1982422, -24978.7636719, 14896.4580078, -39944.0781250, 39899.9531250

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7215155, upper bound: 19342.7134211
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7091618, upper bound: 19342.7082326
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -17028.9472656, 12740.8222656, -19533.1210938, 14727.0341797, -31704.4472656, 32273.9433594
1: -13783.3916016, 12333.3359375, -15808.9785156, 14260.6083984, -28018.3750000, 28142.3105469
2: -20156.9628906, 13454.8154297, -23164.1621094, 15546.1416016, -35668.6171875, 36618.9765625
3: -6984.7680664, 19020.7187500, -8033.3149414, 21906.6015625, -28874.4082031, 27027.5371094
4: -22146.6132812, 13282.3027344, -25439.3359375, 15325.4531250, -37443.3398438, 38721.6406250

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7285409, upper bound: 19342.8591835
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7160343, upper bound: 19342.8449005
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -17028.9472656, 12740.8222656, -21769.1367188, 16309.0244141, -33275.3437500, 34509.9609375
1: -13783.3916016, 12333.3359375, -17622.4628906, 15791.5576172, -29543.3085938, 29955.7968750
2: -20156.9628906, 13454.8154297, -25817.1386719, 17213.8867188, -37326.3984375, 39271.9531250
3: -6984.7680664, 19020.7187500, -8916.7792969, 24385.1230469, -31344.1171875, 27898.7382812
4: -22146.6132812, 13282.3027344, -28349.9394531, 16971.6445312, -39077.7187500, 41632.2421875

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7248966, upper bound: 19342.8118426
time: 4.50 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7250329, upper bound: 19342.8042577
time: 0.94 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -19258.6230469, 14314.8486328, -19533.1210938, 14727.0341797, -33903.2773438, 33847.9687500
1: -15591.9550781, 13858.5380859, -15808.9785156, 14260.6083984, -29802.6523438, 29667.5136719
2: -22803.1542969, 15116.4033203, -23164.1621094, 15546.1416016, -38283.2851562, 38280.5664062
3: -7863.4697266, 21490.8242188, -8033.3149414, 21906.6015625, -29742.3808594, 29488.3125000
4: -25047.6210938, 14921.1982422, -25439.3359375, 15325.4531250, -40309.6992188, 40360.5351562

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7230236, upper bound: 19342.7134211
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7115597, upper bound: 19342.7082326
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -19258.6230469, 14314.8486328, -21769.1367188, 16309.0244141, -35474.1796875, 36083.9843750
1: -15591.9550781, 13858.5380859, -17622.4628906, 15791.5576172, -31327.5878906, 31481.0000000
2: -22803.1542969, 15116.4033203, -25817.1386719, 17213.8867188, -39941.0664062, 40933.5429688
3: -7863.4697266, 21490.8242188, -8916.7792969, 24385.1230469, -32212.0898438, 30359.5136719
4: -25047.6210938, 14921.1982422, -28349.9394531, 16971.6445312, -41944.0781250, 43271.1367188

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7230236, upper bound: 19342.7134211
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7115597, upper bound: 19342.7082326
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -16351.5117188, 12241.2968750, -14998.1894531, 11093.4384766, -27444.9492188, 27239.4863281
1: -13230.7705078, 11842.6308594, -12140.3447266, 10743.0820312, -23973.8496094, 23982.9707031
2: -19316.7578125, 12929.6152344, -17707.3925781, 11735.6181641, -31052.3730469, 30637.0078125
3: -6731.5532227, 18266.6347656, -6128.5795898, 16679.8183594, -23411.3710938, 24395.2128906
4: -21226.9355469, 12769.4882812, -19456.2363281, 11601.2041016, -32828.1406250, 32225.7246094

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_B1_A1_B1_A1_A1

### Relational analysis result of NS_A2_B1_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8695451, upper bound: 19342.8159370
time: 0.96 seconds

## Relational analysis of NS_A2_B1_B1_A1_B1_A1_A2

### Relational analysis result of NS_A2_B1_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8546098, upper bound: 19342.8003056
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -18572.7890625, 13806.0390625, -14998.1894531, 11093.4384766, -29666.2265625, 28804.2285156
1: -15031.4326172, 13358.7333984, -12140.3447266, 10743.0820312, -25774.5097656, 25499.0781250
2: -21952.7890625, 14581.8896484, -17707.3925781, 11735.6181641, -33688.4062500, 32289.2812500
3: -7607.4916992, 20723.9589844, -6128.5795898, 16679.8183594, -24287.3105469, 26852.5351562
4: -24118.3652344, 14400.0126953, -19456.2363281, 11601.2041016, -35719.5664062, 33856.2421875

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8213629, upper bound: 19342.8181928
time: 1.00 seconds

## Relational analysis of NS_A2_B1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8146781, upper bound: 19342.8195326
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -16351.5117188, 12241.2968750, -17222.2578125, 12657.5898438, -29009.1015625, 29463.5546875
1: -13230.7705078, 11842.6308594, -13942.8671875, 12259.5830078, -25490.3535156, 25785.4941406
2: -19316.7578125, 12929.6152344, -20346.2148438, 13389.2744141, -32706.0312500, 33275.8281250
3: -6731.5532227, 18266.6347656, -7004.4199219, 19141.8671875, -25873.4199219, 25271.0546875
4: -21226.9355469, 12769.4882812, -22349.6875000, 13232.6484375, -34459.5820312, 35119.1757812

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_B1_A1_B2_A1_A1

### Relational analysis result of NS_A2_B1_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7270787, upper bound: 19342.8111708
time: 0.75 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2_A1_A2

### Relational analysis result of NS_A2_B1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7233665, upper bound: 19342.7980017
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -18572.7890625, 13806.0390625, -17222.2578125, 12657.5898438, -31230.3789062, 31028.2968750
1: -15031.4326172, 13358.7333984, -13942.8671875, 12259.5830078, -27291.0136719, 27301.5976562
2: -21952.7890625, 14581.8896484, -20346.2148438, 13389.2744141, -35342.0625000, 34928.1054688
3: -7607.4916992, 20723.9589844, -7004.4199219, 19141.8671875, -26749.3574219, 27728.3789062
4: -24118.3652344, 14400.0126953, -22349.6875000, 13232.6484375, -37351.0156250, 36749.6992188

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_B1_A1_B2_A2_A1

### Relational analysis result of NS_A2_B1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7270787, upper bound: 19342.8111710
time: 1.61 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2_A2_A2

### Relational analysis result of NS_A2_B1_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7233665, upper bound: 19342.7980017
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -17101.7656250, 12815.1484375, -15079.4121094, 11150.6660156, -28252.4316406, 27894.5605469
1: -13842.2177734, 12408.0615234, -12206.4462891, 10800.2343750, -24642.4531250, 24614.5019531
2: -20245.4765625, 13537.8974609, -17805.7636719, 11797.1044922, -32042.5781250, 31343.6601562
3: -7023.0063477, 19121.0957031, -6159.3442383, 16770.0722656, -23793.0761719, 25280.4394531
4: -22242.7949219, 13355.7001953, -19564.1445312, 11660.5947266, -33903.3906250, 32919.8437500

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B1_A2_A1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7435816, upper bound: 19342.7442405
time: 0.84 seconds

## Relational analysis of NS_A2_B1_B1_A2_A1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7435816, upper bound: 19342.7442405
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -16963.5136719, 12709.8134766, -17304.0332031, 12715.2548828, -29678.7636719, 30013.8476562
1: -13730.8281250, 12305.0019531, -14009.4804688, 12317.1445312, -26047.9707031, 26314.4824219
2: -20084.6816406, 13425.1523438, -20445.1972656, 13451.1572266, -33535.8398438, 33870.3515625
3: -6963.6406250, 18967.1914062, -7035.3032227, 19232.8066406, -26196.4472656, 26002.4941406
4: -22065.5898438, 13244.2460938, -22458.2949219, 13292.4462891, -35358.0351562, 35702.5390625

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B1_A2_A1_B2_A1

### Relational analysis result of NS_A2_B1_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7435816, upper bound: 19342.7442405
time: 0.72 seconds

## Relational analysis of NS_A2_B1_B1_A2_A1_B2_A2

### Relational analysis result of NS_A2_B1_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7435816, upper bound: 19342.7442405
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -19656.3281250, 14818.3369141, -15079.4121094, 11150.6660156, -30806.9941406, 29865.4355469
1: -15908.4462891, 14347.5273438, -12206.4462891, 10800.2343750, -26708.6796875, 26542.9453125
2: -23309.0292969, 15641.6455078, -17805.7636719, 11797.1044922, -35106.1250000, 33447.4101562
3: -8083.9985352, 22043.9179688, -6159.3442383, 16770.0722656, -24854.0703125, 28203.2617188
4: -25598.7207031, 15420.4580078, -19564.1445312, 11660.5947266, -37259.3164062, 34984.6015625

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B1_A2_A2_B1_A1

### Relational analysis result of NS_A2_B1_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7530863, upper bound: 19342.8236627
time: 1.17 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7530863, upper bound: 19342.8236626
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -19510.4570312, 14707.0664062, -17304.0332031, 12715.2548828, -32225.7109375, 31942.7382812
1: -15791.1826172, 14239.3886719, -14009.4804688, 12317.1445312, -28108.3261719, 28209.3398438
2: -23140.1054688, 15523.1376953, -20445.1972656, 13451.1572266, -36591.2578125, 35933.2539062
3: -8021.8105469, 21882.7363281, -7035.3032227, 19232.8066406, -27246.5527344, 28918.0390625
4: -25412.6054688, 15302.9814453, -22458.2949219, 13292.4462891, -38705.0507812, 37724.6210938

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B1_A2_A2_B2_A1

### Relational analysis result of NS_A2_B1_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7530863, upper bound: 19342.8236626
time: 0.88 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2_B2_A2

### Relational analysis result of NS_A2_B1_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7530863, upper bound: 19342.8236627
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -16960.5136719, 12710.3867188, -16960.5136719, 12710.3867188, -29670.9003906, 29670.9003906
1: -13727.9248047, 12307.4902344, -13727.9248047, 12307.4902344, -26035.4140625, 26035.4140625
2: -20078.0742188, 13428.0615234, -20078.0742188, 13428.0615234, -33506.1328125, 33506.1367188
3: -6965.6933594, 18962.8554688, -6965.6933594, 18962.8554688, -25928.5488281, 25928.5488281
4: -22058.8867188, 13246.7812500, -22058.8867188, 13246.7812500, -35305.6679688, 35305.6679688

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_B2_A1_A1_B1_B1

### Relational analysis result of NS_A2_B1_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7255165, upper bound: 19342.8539092
time: 0.79 seconds

## Relational analysis of NS_A2_B1_B2_A1_A1_B1_B2

### Relational analysis result of NS_A2_B1_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7107827, upper bound: 19342.8387908
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -16960.5136719, 12710.3867188, -19204.0390625, 14295.5185547, -31256.0312500, 31914.4257812
1: -13727.9248047, 12307.4902344, -15547.9160156, 13842.3193359, -27570.2421875, 27855.4042969
2: -20078.0742188, 13428.0615234, -22741.0097656, 15099.6142578, -35177.6875000, 36169.0703125
3: -6965.6933594, 18962.8554688, -7851.1518555, 21447.9667969, -28413.6601562, 26814.0058594
4: -22058.8867188, 13246.7812500, -24978.7636719, 14896.4580078, -36955.3437500, 38225.5468750

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_B2_A1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7156758, upper bound: 19342.8406111
time: 0.85 seconds

## Relational analysis of NS_A2_B1_B2_A1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7107827, upper bound: 19342.8387908
time: 3.06 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -19204.0390625, 14295.5185547, -16960.5136719, 12710.3867188, -31914.4257812, 31256.0312500
1: -15547.9160156, 13842.3193359, -13727.9248047, 12307.4902344, -27855.4062500, 27570.2421875
2: -22741.0097656, 15099.6142578, -20078.0742188, 13428.0615234, -36169.0703125, 35177.6835938
3: -7851.1518555, 21447.9667969, -6965.6933594, 18962.8554688, -26814.0078125, 28413.6601562
4: -24978.7636719, 14896.4580078, -22058.8867188, 13246.7812500, -38225.5468750, 36955.3437500

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_B2_A1_A2_B1_B1

### Relational analysis result of NS_A2_B1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7442749, upper bound: 19342.6782303
time: 0.97 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_B2_A1_A2_B1_B1

### Relational analysis result of NS_A2_B1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7216044, upper bound: 19342.7141133
time: 0.92 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7090545, upper bound: 19342.7090524
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -19204.0390625, 14295.5185547, -19204.0390625, 14295.5185547, -33499.5585938, 33499.5585938
1: -15547.9160156, 13842.3193359, -15547.9160156, 13842.3193359, -29390.2343750, 29390.2304688
2: -22741.0097656, 15099.6142578, -22741.0097656, 15099.6142578, -37840.6210938, 37840.6250000
3: -7851.1518555, 21447.9667969, -7851.1518555, 21447.9667969, -29299.1152344, 29299.1171875
4: -24978.7636719, 14896.4580078, -24978.7636719, 14896.4580078, -39875.2187500, 39875.2148438

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_B2_A1_A2_B2_B1

### Relational analysis result of NS_A2_B1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7216044, upper bound: 19342.7141133
time: 0.86 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2_B2_B2

### Relational analysis result of NS_A2_B1_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7090545, upper bound: 19342.7090524
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -19533.1210938, 14727.0341797, -16960.5136719, 12710.3867188, -32243.5078125, 31642.1542969
1: -15808.9785156, 14260.6083984, -13727.9248047, 12307.4902344, -28116.4687500, 27966.5898438
2: -23164.1621094, 15546.1416016, -20078.0742188, 13428.0615234, -36592.2226562, 35594.0585938
3: -8033.3149414, 21906.6015625, -6965.6933594, 18962.8554688, -26968.0214844, 28851.1816406
4: -25439.3359375, 15325.4531250, -22058.8867188, 13246.7812500, -38686.1171875, 37363.0585938

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8161476, upper bound: 19342.8214755
time: 0.85 seconds

## Relational analysis of NS_A2_B1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8107805, upper bound: 19342.8222641
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -21769.1367188, 16309.0244141, -16960.5136719, 12710.3867188, -34479.5234375, 33213.0546875
1: -17622.4628906, 15791.5576172, -13727.9248047, 12307.4902344, -29929.9531250, 29491.5253906
2: -25817.1386719, 17213.8867188, -20078.0742188, 13428.0615234, -39245.1992188, 37251.8359375
3: -8916.7792969, 24385.1230469, -6965.6933594, 18962.8554688, -27839.2226562, 31320.8886719
4: -28349.9394531, 16971.6445312, -22058.8867188, 13246.7812500, -41596.7187500, 38997.4414062

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8161476, upper bound: 19342.8214755
time: 0.82 seconds

## Relational analysis of NS_A2_B1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8107805, upper bound: 19342.8222647
time: 4.54 seconds

## BFS NS instance: NS_A2_B1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -19533.1210938, 14727.0341797, -19204.0390625, 14295.5185547, -33828.6406250, 33854.5625000
1: -15808.9785156, 14260.6083984, -15547.9160156, 13842.3193359, -29651.2968750, 29762.0000000
2: -23164.1621094, 15546.1416016, -22741.0097656, 15099.6142578, -38263.7734375, 38225.1367188
3: -8033.3149414, 21906.6015625, -7851.1518555, 21447.9667969, -29444.1503906, 29724.0527344
4: -25439.3359375, 15325.4531250, -24978.7636719, 14896.4580078, -40335.7929688, 40247.9648438

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_B2_A2_B2_A1_A1

### Relational analysis result of NS_A2_B1_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7277790, upper bound: 19342.8137082
time: 0.91 seconds

## Relational analysis of NS_A2_B1_B2_A2_B2_A1_A2

### Relational analysis result of NS_A2_B1_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7240663, upper bound: 19342.7999006
time: 1.77 seconds

## BFS NS instance: NS_A2_B1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -21769.1367188, 16309.0244141, -19204.0390625, 14295.5185547, -36064.6562500, 35425.4687500
1: -17622.4628906, 15791.5576172, -15547.9160156, 13842.3193359, -31464.7792969, 31286.9335938
2: -25817.1386719, 17213.8867188, -22741.0097656, 15099.6142578, -40914.4335938, 39882.9101562
3: -8916.7792969, 24385.1230469, -7851.1518555, 21447.9667969, -30315.3515625, 32193.7617188
4: -28349.9394531, 16971.6445312, -24978.7636719, 14896.4580078, -43246.3984375, 41882.3476562

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_B2_A2_B2_A2_A1

### Relational analysis result of NS_A2_B1_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7277790, upper bound: 19342.8137082
time: 0.81 seconds

## Relational analysis of NS_A2_B1_B2_A2_B2_A2_A2

### Relational analysis result of NS_A2_B1_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7240663, upper bound: 19342.7998999
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -16960.5136719, 12710.3867188, -17028.9472656, 12740.8222656, -29701.3359375, 29739.3320312
1: -13727.9248047, 12307.4902344, -13783.3916016, 12333.3359375, -26061.2578125, 26090.8828125
2: -20078.0742188, 13428.0615234, -20156.9628906, 13454.8154297, -33532.8867188, 33585.0234375
3: -6965.6933594, 18962.8554688, -6984.7680664, 19020.7187500, -25986.4121094, 25947.6230469
4: -22058.8867188, 13246.7812500, -22146.6132812, 13282.3027344, -35341.1835938, 35393.3945312

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B1_A1_B1_A1_A1

### Relational analysis result of NS_A2_B2_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8583178, upper bound: 19342.7266415
time: 0.83 seconds

## Relational analysis of NS_A2_B2_B1_A1_B1_A1_A2

### Relational analysis result of NS_A2_B2_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8441606, upper bound: 19342.7120750
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -19204.0390625, 14295.5185547, -17028.9472656, 12740.8222656, -31944.8613281, 31324.4648438
1: -15547.9160156, 13842.3193359, -13783.3916016, 12333.3359375, -27881.2460938, 27625.7109375
2: -22741.0097656, 15099.6142578, -20156.9628906, 13454.8154297, -36195.8242188, 35256.5781250
3: -7851.1518555, 21447.9667969, -6984.7680664, 19020.7187500, -26871.8710938, 28432.7324219
4: -24978.7636719, 14896.4580078, -22146.6132812, 13282.3027344, -38261.0625000, 37043.0703125

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8441606, upper bound: 19342.7159401
time: 0.80 seconds

## Relational analysis of NS_A2_B2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8441606, upper bound: 19342.7120750
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -16960.5136719, 12710.3867188, -19258.6230469, 14314.8486328, -31275.3632812, 31969.0097656
1: -13727.9248047, 12307.4902344, -15591.9550781, 13858.5380859, -27586.4609375, 27899.4453125
2: -20078.0742188, 13428.0615234, -22803.1542969, 15116.4033203, -35194.4765625, 36231.2148438
3: -6965.6933594, 18962.8554688, -7863.4697266, 21490.8242188, -28456.5175781, 26826.3242188
4: -22058.8867188, 13246.7812500, -25047.6210938, 14921.1982422, -36980.0859375, 38294.4023438

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B1_A1_B2_A1_A1

### Relational analysis result of NS_A2_B2_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7134211, upper bound: 19342.7215155
time: 0.93 seconds

## Relational analysis of NS_A2_B2_B1_A1_B2_A1_A2

### Relational analysis result of NS_A2_B2_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7082325, upper bound: 19342.7091618
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -19204.0390625, 14295.5185547, -19258.6230469, 14314.8486328, -33518.8867188, 33554.1406250
1: -15547.9160156, 13842.3193359, -15591.9550781, 13858.5380859, -29406.4511719, 29434.2734375
2: -22741.0097656, 15099.6142578, -22803.1542969, 15116.4033203, -37857.4140625, 37902.7695312
3: -7851.1518555, 21447.9667969, -7863.4697266, 21490.8242188, -29341.9746094, 29311.4335938
4: -24978.7636719, 14896.4580078, -25047.6210938, 14921.1982422, -39899.9531250, 39944.0781250

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B1_A1_B2_A2_A1

### Relational analysis result of NS_A2_B2_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7134211, upper bound: 19342.7215154
time: 0.84 seconds

## Relational analysis of NS_A2_B2_B1_A1_B2_A2_A2

### Relational analysis result of NS_A2_B2_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7082326, upper bound: 19342.7091618
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -19533.1210938, 14727.0341797, -17028.9472656, 12740.8222656, -32273.9433594, 31704.4453125
1: -15808.9785156, 14260.6083984, -13783.3916016, 12333.3359375, -28142.3125000, 28018.3730469
2: -23164.1621094, 15546.1416016, -20156.9628906, 13454.8154297, -36618.9765625, 35668.6171875
3: -8033.3149414, 21906.6015625, -6984.7680664, 19020.7187500, -27027.5371094, 28874.4101562
4: -25439.3359375, 15325.4531250, -22146.6132812, 13282.3027344, -38721.6406250, 37443.3398438

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B1_A2_B1_A1_A1

### Relational analysis result of NS_A2_B2_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8696383, upper bound: 19342.8175749
time: 0.98 seconds

## Relational analysis of NS_A2_B2_B1_A2_B1_A1_A2

### Relational analysis result of NS_A2_B2_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8546345, upper bound: 19342.8035721
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -21769.1367188, 16309.0244141, -17028.9472656, 12740.8222656, -34509.9609375, 33275.3398438
1: -17622.4628906, 15791.5576172, -13783.3916016, 12333.3359375, -29955.7988281, 29543.3085938
2: -25817.1386719, 17213.8867188, -20156.9628906, 13454.8154297, -39271.9531250, 37326.3945312
3: -8916.7792969, 24385.1230469, -6984.7680664, 19020.7187500, -27898.7382812, 31344.1171875
4: -28349.9394531, 16971.6445312, -22146.6132812, 13282.3027344, -41632.2421875, 39077.7187500

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8226594, upper bound: 19342.8185593
time: 0.88 seconds

## Relational analysis of NS_A2_B2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8161769, upper bound: 19342.8204367
time: 1.64 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -19533.1210938, 14727.0341797, -19258.6230469, 14314.8486328, -33847.9687500, 33903.2773438
1: -15808.9785156, 14260.6083984, -15591.9550781, 13858.5380859, -29667.5136719, 29802.6523438
2: -23164.1621094, 15546.1416016, -22803.1542969, 15116.4033203, -38280.5664062, 38283.2890625
3: -8033.3149414, 21906.6015625, -7863.4697266, 21490.8242188, -29488.3164062, 29742.3789062
4: -25439.3359375, 15325.4531250, -25047.6210938, 14921.1982422, -40360.5351562, 40309.6992188

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B1_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7269680, upper bound: 19342.8131673
time: 0.76 seconds

## Relational analysis of NS_A2_B2_B1_A2_B2_A1_A2

### Relational analysis result of NS_A2_B2_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7228720, upper bound: 19342.8006299
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -21769.1367188, 16309.0244141, -19258.6230469, 14314.8486328, -36083.9843750, 35474.1796875
1: -17622.4628906, 15791.5576172, -15591.9550781, 13858.5380859, -31481.0000000, 31327.5878906
2: -25817.1386719, 17213.8867188, -22803.1542969, 15116.4033203, -40933.5429688, 39941.0625000
3: -8916.7792969, 24385.1230469, -7863.4697266, 21490.8242188, -30359.5136719, 32212.0898438
4: -28349.9394531, 16971.6445312, -25047.6210938, 14921.1982422, -43271.1367188, 41944.0781250

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B1_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7269680, upper bound: 19342.8131670
time: 0.89 seconds

## Relational analysis of NS_A2_B2_B1_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7228720, upper bound: 19342.8006303
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -16960.5136719, 12710.3867188, -19533.1210938, 14727.0341797, -31642.1542969, 32243.5078125
1: -13727.9248047, 12307.4902344, -15808.9785156, 14260.6083984, -27966.5917969, 28116.4687500
2: -20078.0742188, 13428.0615234, -23164.1621094, 15546.1416016, -35594.0585938, 36592.2226562
3: -6965.6933594, 18962.8554688, -8033.3149414, 21906.6015625, -28851.1796875, 26968.0234375
4: -22058.8867188, 13246.7812500, -25439.3359375, 15325.4531250, -37363.0585938, 38686.1171875

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_B2_A1_A1_B1_A1

### Relational analysis result of NS_A2_B2_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7287786, upper bound: 19342.8016813
time: 0.92 seconds

## Relational analysis of NS_A2_B2_B2_A1_A1_B1_A2

### Relational analysis result of NS_A2_B2_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7288545, upper bound: 19342.7969150
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -16960.5136719, 12710.3867188, -21769.1367188, 16309.0244141, -33213.0546875, 34479.5234375
1: -13727.9248047, 12307.4902344, -17622.4628906, 15791.5576172, -29491.5253906, 29929.9531250
2: -20078.0742188, 13428.0615234, -25817.1386719, 17213.8867188, -37251.8359375, 39245.1992188
3: -6965.6933594, 18962.8554688, -8916.7792969, 24385.1230469, -31320.8886719, 27839.2226562
4: -22058.8867188, 13246.7812500, -28349.9394531, 16971.6445312, -38997.4414062, 41596.7187500

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_B2_A1_A1_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7287786, upper bound: 19342.8016813
time: 0.78 seconds

## Relational analysis of NS_A2_B2_B2_A1_A1_B2_A2

### Relational analysis result of NS_A2_B2_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7288545, upper bound: 19342.7969150
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -19204.0390625, 14295.5185547, -19533.1210938, 14727.0341797, -33854.5625000, 33828.6406250
1: -15547.9160156, 13842.3193359, -15808.9785156, 14260.6083984, -29761.9960938, 29651.2968750
2: -22741.0097656, 15099.6142578, -23164.1621094, 15546.1416016, -38225.1367188, 38263.7734375
3: -7851.1518555, 21447.9667969, -8033.3149414, 21906.6015625, -29724.0546875, 29444.1503906
4: -24978.7636719, 14896.4580078, -25439.3359375, 15325.4531250, -40247.9648438, 40335.7929688

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B2_A1_A2_B1_B1

### Relational analysis result of NS_A2_B2_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7205848, upper bound: 19342.7136173
time: 0.85 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2_B1_B2

### Relational analysis result of NS_A2_B2_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7108658, upper bound: 19342.7097567
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -19204.0390625, 14295.5185547, -21769.1367188, 16309.0244141, -35425.4687500, 36064.6562500
1: -15547.9160156, 13842.3193359, -17622.4628906, 15791.5576172, -31286.9335938, 31464.7792969
2: -22741.0097656, 15099.6142578, -25817.1386719, 17213.8867188, -39882.9140625, 40914.4335938
3: -7851.1518555, 21447.9667969, -8916.7792969, 24385.1230469, -32193.7617188, 30315.3515625
4: -24978.7636719, 14896.4580078, -28349.9394531, 16971.6445312, -41882.3437500, 43246.3984375

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_B1

### Relational analysis result of NS_A2_B2_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7205848, upper bound: 19342.7136173
time: 0.81 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7108658, upper bound: 19342.7097567
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -19533.1210938, 14727.0341797, -19533.1210938, 14727.0341797, -34172.5898438, 34172.5898438
1: -15808.9785156, 14260.6083984, -15808.9785156, 14260.6083984, -30014.5429688, 30014.5429688
2: -23164.1621094, 15546.1416016, -23164.1621094, 15546.1416016, -38614.4296875, 38614.4335938
3: -8033.3149414, 21906.6015625, -8033.3149414, 21906.6015625, -29863.0585938, 29863.0566406
4: -25439.3359375, 15325.4531250, -25439.3359375, 15325.4531250, -40683.1132812, 40683.1132812

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B2_A2_A1_B1_A1

### Relational analysis result of NS_A2_B2_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8249605, upper bound: 19342.9408773
time: 1.18 seconds

## Relational analysis of NS_A2_B2_B2_A2_A1_B1_A2

### Relational analysis result of NS_A2_B2_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8228000, upper bound: 19342.9362779
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -19533.1210938, 14727.0341797, -21769.1367188, 16309.0244141, -35743.5000000, 36380.8007812
1: -15808.9785156, 14260.6083984, -17622.4628906, 15791.5576172, -31539.4765625, 31805.9550781
2: -23164.1621094, 15546.1416016, -25817.1386719, 17213.8867188, -40272.2148438, 41238.8359375
3: -8033.3149414, 21906.6015625, -8916.7792969, 24385.1230469, -32332.7675781, 30734.2597656
4: -25439.3359375, 15325.4531250, -28349.9394531, 16971.6445312, -42317.4921875, 43561.9296875

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B2_A2_A1_B2_A1

### Relational analysis result of NS_A2_B2_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8249605, upper bound: 19342.9408774
time: 0.74 seconds

## Relational analysis of NS_A2_B2_B2_A2_A1_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8228000, upper bound: 19342.9362779
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -21769.1367188, 16309.0244141, -19533.1210938, 14727.0341797, -36380.8046875, 35743.4960938
1: -17622.4628906, 15791.5576172, -15808.9785156, 14260.6083984, -31805.9589844, 31539.4765625
2: -25817.1386719, 17213.8867188, -23164.1621094, 15546.1416016, -41238.8359375, 40272.2148438
3: -8916.7792969, 24385.1230469, -8033.3149414, 21906.6015625, -30734.2558594, 32332.7675781
4: -28349.9394531, 16971.6445312, -25439.3359375, 15325.4531250, -43561.9335938, 42317.4921875

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B2_A2_A2_B1_B1

### Relational analysis result of NS_A2_B2_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8297160, upper bound: 19342.8212345
time: 0.92 seconds

## Relational analysis of NS_A2_B2_B2_A2_A2_B1_B2

### Relational analysis result of NS_A2_B2_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8185624, upper bound: 19342.8185521
time: 0.98 seconds

## BFS NS instance: NS_A2_B2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -21769.1367188, 16309.0244141, -21769.1367188, 16309.0244141, -37951.7031250, 37951.7070312
1: -17622.4628906, 15791.5576172, -17622.4628906, 15791.5576172, -33330.8945312, 33330.8945312
2: -25817.1386719, 17213.8867188, -25817.1386719, 17213.8867188, -42896.6171875, 42896.6171875
3: -8916.7792969, 24385.1230469, -8916.7792969, 24385.1230469, -33203.9648438, 33203.9648438
4: -28349.9394531, 16971.6445312, -28349.9394531, 16971.6445312, -45196.3125000, 45196.3125000

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B2_A2_A2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8210597, upper bound: 19342.8295866
time: 0.89 seconds

## Relational analysis of NS_A2_B2_B2_A2_A2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8185624, upper bound: 19342.8185521
time: 0.81 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 2.80 seconds
NS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7157609, upper bound: 19342.8435579
NS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7109404, upper bound: 19342.8435579
NS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19340.5577195, upper bound: 19342.8392300
NS_A1_B1_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.80
Output dim: 3, lower bound: -19340.4319644, upper bound: 19340.5482198
NS_A1_B1_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7397225, upper bound: 19342.6776208
NS_A1_B1_A1_B1_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 2.80
Output dim: 3, lower bound: -19340.5482198, upper bound: 19340.9681810
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7134936, upper bound: 19342.7190485
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7085000, upper bound: 19342.7085000
NS_A1_B1_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7181726, upper bound: 19342.8600947
NS_A1_B1_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7107537, upper bound: 19342.8436747
NS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7153835, upper bound: 19342.8103019
NS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7168218, upper bound: 19342.8026693
NS_A1_B1_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7135128, upper bound: 19342.7125960
NS_A1_B1_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7081476, upper bound: 19342.7088192
NS_A1_B1_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7135128, upper bound: 19342.7125960
NS_A1_B1_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7081476, upper bound: 19342.7088192
NS_A1_B1_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8600947, upper bound: 19342.7181726
NS_A1_B1_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8436747, upper bound: 19342.7107537
NS_A1_B1_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8103019, upper bound: 19342.7153835
NS_A1_B1_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8026693, upper bound: 19342.7168218
NS_A1_B1_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7125960, upper bound: 19342.7135128
NS_A1_B1_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7088192, upper bound: 19342.7081476
NS_A1_B1_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7125960, upper bound: 19342.7135128
NS_A1_B1_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7088192, upper bound: 19342.7081476
NS_A1_B1_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8580181, upper bound: 19342.7181726
NS_A1_B1_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8428644, upper bound: 19342.7107537
NS_A1_B1_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8428761, upper bound: 19342.7157363
NS_A1_B1_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8428644, upper bound: 19342.7107537
NS_A1_B1_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7125960, upper bound: 19342.7135128
NS_A1_B1_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7083976, upper bound: 19342.7081476
NS_A1_B1_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7125960, upper bound: 19342.7135128
NS_A1_B1_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7083976, upper bound: 19342.7081476
NS_A1_B2_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8159368, upper bound: 19342.8695451
NS_A1_B2_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8003056, upper bound: 19342.8546098
NS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8181931, upper bound: 19342.8213629
NS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8195324, upper bound: 19342.8146781
NS_A1_B2_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8111708, upper bound: 19342.7270787
NS_A1_B2_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7980017, upper bound: 19342.7233665
NS_A1_B2_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8111712, upper bound: 19342.7270787
NS_A1_B2_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7980017, upper bound: 19342.7233665
NS_A1_B2_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7442405, upper bound: 19342.7435816
NS_A1_B2_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7442405, upper bound: 19342.7435816
NS_A1_B2_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7442405, upper bound: 19342.7435816
NS_A1_B2_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7442405, upper bound: 19342.7435816
NS_A1_B2_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8236629, upper bound: 19342.7530863
NS_A1_B2_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8236629, upper bound: 19342.7530863
NS_A1_B2_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8236626, upper bound: 19342.7530863
NS_A1_B2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8236626, upper bound: 19342.7530863
NS_A1_B2_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7266415, upper bound: 19342.8583178
NS_A1_B2_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7120750, upper bound: 19342.8441606
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7159401, upper bound: 19342.8441606
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7120750, upper bound: 19342.8441606
NS_A1_B2_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7191075, upper bound: 19342.7134211
NS_A1_B2_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7091618, upper bound: 19342.7082326
NS_A1_B2_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7215155, upper bound: 19342.7134211
NS_A1_B2_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7091618, upper bound: 19342.7082326
NS_A1_B2_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7285409, upper bound: 19342.8591835
NS_A1_B2_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7160343, upper bound: 19342.8449005
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7248966, upper bound: 19342.8118426
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7250329, upper bound: 19342.8042577
NS_A1_B2_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7230236, upper bound: 19342.7134211
NS_A1_B2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7115597, upper bound: 19342.7082326
NS_A1_B2_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7230236, upper bound: 19342.7134211
NS_A1_B2_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7115597, upper bound: 19342.7082326
NS_A2_B1_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8695451, upper bound: 19342.8159370
NS_A2_B1_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8546098, upper bound: 19342.8003056
NS_A2_B1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8213629, upper bound: 19342.8181928
NS_A2_B1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8146781, upper bound: 19342.8195326
NS_A2_B1_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7270787, upper bound: 19342.8111708
NS_A2_B1_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7233665, upper bound: 19342.7980017
NS_A2_B1_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7270787, upper bound: 19342.8111710
NS_A2_B1_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7233665, upper bound: 19342.7980017
NS_A2_B1_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7435816, upper bound: 19342.7442405
NS_A2_B1_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7435816, upper bound: 19342.7442405
NS_A2_B1_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7435816, upper bound: 19342.7442405
NS_A2_B1_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7435816, upper bound: 19342.7442405
NS_A2_B1_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7530863, upper bound: 19342.8236627
NS_A2_B1_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7530863, upper bound: 19342.8236626
NS_A2_B1_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7530863, upper bound: 19342.8236626
NS_A2_B1_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7530863, upper bound: 19342.8236627
NS_A2_B1_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7255165, upper bound: 19342.8539092
NS_A2_B1_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7107827, upper bound: 19342.8387908
NS_A2_B1_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7156758, upper bound: 19342.8406111
NS_A2_B1_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7107827, upper bound: 19342.8387908
NS_A2_B1_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7216044, upper bound: 19342.7141133
NS_A2_B1_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7090545, upper bound: 19342.7090524
NS_A2_B1_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7216044, upper bound: 19342.7141133
NS_A2_B1_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7090545, upper bound: 19342.7090524
NS_A2_B1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8161476, upper bound: 19342.8214755
NS_A2_B1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8107805, upper bound: 19342.8222641
NS_A2_B1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8161476, upper bound: 19342.8214755
NS_A2_B1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8107805, upper bound: 19342.8222647
NS_A2_B1_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7277790, upper bound: 19342.8137082
NS_A2_B1_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7240663, upper bound: 19342.7999006
NS_A2_B1_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7277790, upper bound: 19342.8137082
NS_A2_B1_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7240663, upper bound: 19342.7998999
NS_A2_B2_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8583178, upper bound: 19342.7266415
NS_A2_B2_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8441606, upper bound: 19342.7120750
NS_A2_B2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8441606, upper bound: 19342.7159401
NS_A2_B2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8441606, upper bound: 19342.7120750
NS_A2_B2_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7134211, upper bound: 19342.7215155
NS_A2_B2_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7082325, upper bound: 19342.7091618
NS_A2_B2_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7134211, upper bound: 19342.7215154
NS_A2_B2_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7082326, upper bound: 19342.7091618
NS_A2_B2_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8696383, upper bound: 19342.8175749
NS_A2_B2_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8546345, upper bound: 19342.8035721
NS_A2_B2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8226594, upper bound: 19342.8185593
NS_A2_B2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8161769, upper bound: 19342.8204367
NS_A2_B2_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7269680, upper bound: 19342.8131673
NS_A2_B2_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7228720, upper bound: 19342.8006299
NS_A2_B2_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7269680, upper bound: 19342.8131670
NS_A2_B2_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7228720, upper bound: 19342.8006303
NS_A2_B2_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7287786, upper bound: 19342.8016813
NS_A2_B2_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7288545, upper bound: 19342.7969150
NS_A2_B2_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7287786, upper bound: 19342.8016813
NS_A2_B2_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7288545, upper bound: 19342.7969150
NS_A2_B2_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7205848, upper bound: 19342.7136173
NS_A2_B2_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7108658, upper bound: 19342.7097567
NS_A2_B2_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7205848, upper bound: 19342.7136173
NS_A2_B2_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.7108658, upper bound: 19342.7097567
NS_A2_B2_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8249605, upper bound: 19342.9408773
NS_A2_B2_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8228000, upper bound: 19342.9362779
NS_A2_B2_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8249605, upper bound: 19342.9408774
NS_A2_B2_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8228000, upper bound: 19342.9362779
NS_A2_B2_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8297160, upper bound: 19342.8212345
NS_A2_B2_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8185624, upper bound: 19342.8185521
NS_A2_B2_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8210597, upper bound: 19342.8295866
NS_A2_B2_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 3, lower bound: -19342.8185624, upper bound: 19342.8185521

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -14852.7773438, 10980.1337891, -14973.5019531, 11070.9960938, -25923.7675781, 25953.6308594
1: -12023.2802734, 10635.0234375, -12120.8447266, 10723.0634766, -22746.3437500, 22755.8671875
2: -17537.8535156, 11616.9316406, -17680.5507812, 11712.9384766, -29250.7910156, 29297.4824219
3: -6065.4174805, 16515.6132812, -6115.4731445, 16651.1816406, -22716.5996094, 22631.0859375
4: -19269.9804688, 11482.8046875, -19426.6660156, 11577.5195312, -30847.5000000, 30909.4707031

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9344730, upper bound: 19342.9344730
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9344730, upper bound: 19342.9344730
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -16946.2988281, 12532.1083984, -14934.1025391, 11041.8251953, -27988.1230469, 27466.2109375
1: -13720.8076172, 12135.0527344, -12088.9697266, 10694.9960938, -24415.8046875, 24224.0234375
2: -20013.1718750, 13255.9873047, -17634.5859375, 11682.4072266, -31695.5781250, 30890.5742188
3: -6922.2622070, 18865.2890625, -6099.0957031, 16607.8964844, -23530.1582031, 24964.3847656
4: -21991.1601562, 13110.0791016, -19376.0117188, 11547.1044922, -33538.2656250, 32486.0898438

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9344730, upper bound: 19342.9344730
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9344730, upper bound: 19342.9344730
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -14723.2949219, 10884.7685547, -17147.2265625, 12596.3515625, -27319.6464844, 28031.9941406
1: -11919.1562500, 10543.6269531, -13882.4335938, 12202.5957031, -24121.7519531, 24426.0605469
2: -17386.8886719, 11516.7431641, -20259.7656250, 13324.8955078, -30711.7812500, 31776.5078125
3: -6012.6723633, 16373.9775391, -6971.4277344, 19057.7402344, -25070.4101562, 23345.4062500
4: -19104.0039062, 11383.1757812, -22254.7695312, 13166.9599609, -32270.9648438, 33637.9453125

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19340.9681810, upper bound: 19340.5482198
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19340.9681810, upper bound: 19340.5482198
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -17147.2265625, 12596.3515625, -14723.2949219, 10884.7685547, -28031.9960938, 27319.6464844
1: -13882.4335938, 12202.5957031, -11919.1562500, 10543.6269531, -24426.0605469, 24121.7519531
2: -20259.7656250, 13324.8955078, -17386.8886719, 11516.7431641, -31776.5078125, 30711.7812500
3: -6971.4277344, 19057.7402344, -6012.6723633, 16373.9775391, -23345.4062500, 25070.4082031
4: -22254.7695312, 13166.9599609, -19104.0039062, 11383.1757812, -33637.9453125, 32270.9648438

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19340.5482198, upper bound: 19340.9681810
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19340.5482198, upper bound: 19340.9681810
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -17082.3515625, 12548.6572266, -17199.3613281, 12636.5185547, -29718.8691406, 29748.0175781
1: -13830.2763672, 12156.1923828, -13924.8984375, 12241.0957031, -26071.3710938, 26081.0898438
2: -20183.1171875, 13275.0468750, -20321.4941406, 13367.9326172, -33551.0507812, 33596.5312500
3: -6943.6918945, 18984.3632812, -6991.9951172, 19115.5136719, -26059.2031250, 25976.3593750
4: -22170.4941406, 13118.4257812, -22322.4218750, 13210.2080078, -35380.6992188, 35440.8476562

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7085000, upper bound: 19342.7085000
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7085000, upper bound: 19342.7085000
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -19149.1718750, 14076.0498047, -17165.0429688, 12611.3037109, -31760.4687500, 31241.0937500
1: -15505.9082031, 13632.9433594, -13897.1591797, 12216.7861328, -27722.6914062, 27530.1015625
2: -22625.4472656, 14888.5498047, -20281.4785156, 13341.2685547, -35966.7109375, 35170.0234375
3: -7787.9921875, 21300.0195312, -6977.6308594, 19078.0195312, -26866.0117188, 28277.6503906
4: -24855.4941406, 14720.6015625, -22278.3281250, 13183.6787109, -38039.1718750, 36998.9218750

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7085000, upper bound: 19342.7085000
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7085000, upper bound: 19342.7085000
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -14973.5019531, 11070.9960938, -16800.3320312, 12569.3457031, -27542.8437500, 27871.3242188
1: -12120.8447266, 10723.0634766, -13598.5332031, 12167.5263672, -24288.3710938, 24321.5976562
2: -17680.5507812, 11712.9384766, -19886.5273438, 13273.5097656, -30954.0605469, 31599.4648438
3: -6115.4731445, 16651.1816406, -6890.1635742, 18764.3359375, -24879.8027344, 23541.3417969
4: -19426.6660156, 11577.5195312, -21849.7050781, 13103.6064453, -32530.2714844, 33427.2226562

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9358136, upper bound: 19342.9364883
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9358136, upper bound: 19342.9364883
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -14934.1025391, 11041.8251953, -18828.3105469, 14064.7089844, -28998.8125000, 29870.1328125
1: -12088.9697266, 10694.9960938, -15242.4082031, 13612.1025391, -25701.0722656, 25937.4042969
2: -17634.5859375, 11682.4072266, -22282.5332031, 14852.5244141, -32487.1074219, 33964.9375000
3: -6099.0957031, 16607.8964844, -7717.3305664, 21036.9863281, -27136.0820312, 24325.2265625
4: -19376.0117188, 11547.1044922, -24483.2949219, 14674.1562500, -34050.1679688, 36030.3984375

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9358136, upper bound: 19342.9364883
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9358136, upper bound: 19342.9364883
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -14857.6435547, 10986.6835938, -19113.0937500, 14205.7099609, -29063.3535156, 30099.7753906
1: -12026.8066406, 10641.2021484, -15473.9306641, 13753.4726562, -25780.2792969, 26115.1328125
2: -17544.5175781, 11624.1933594, -22631.7929688, 15001.4619141, -32545.9765625, 34255.9843750
3: -6069.5864258, 16522.9160156, -7804.1357422, 21328.9472656, -27398.5332031, 24327.0507812
4: -19277.2167969, 11489.2861328, -24859.4785156, 14807.2119141, -34084.4257812, 36348.7656250

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7084733, upper bound: 19342.8047188
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.6926236, upper bound: 19342.7928918
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -15604.7080078, 11584.0917969, -18980.8945312, 14112.5683594, -29717.2773438, 30564.9863281
1: -12633.4902344, 11221.5693359, -15367.6513672, 13664.3476562, -26297.8339844, 26589.2207031
2: -18432.1875000, 12250.5087891, -22478.7167969, 14904.7080078, -33336.8945312, 34729.2226562
3: -6392.2875977, 17400.9101562, -7751.0849609, 21184.7324219, -27577.0195312, 25151.9941406
4: -20245.6445312, 12099.2636719, -24688.8906250, 14710.4316406, -34956.0664062, 36788.1523438

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7084733, upper bound: 19342.7971368
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.6926435, upper bound: 19342.7911924
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -17199.3613281, 12636.5185547, -16800.3320312, 12569.3457031, -29768.7070312, 29436.8515625
1: -13924.8984375, 12241.0957031, -13598.5332031, 12167.5263672, -26092.4238281, 25839.6289062
2: -20321.4941406, 13367.9326172, -19886.5273438, 13273.5097656, -33594.9960938, 33254.4609375
3: -6991.9951172, 19115.5136719, -6890.1635742, 18764.3359375, -25756.3300781, 26005.6738281
4: -22322.4218750, 13210.2080078, -21849.7050781, 13103.6064453, -35426.0273438, 35059.9140625

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8433286, upper bound: 19342.7119489
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8433286, upper bound: 19342.7119489
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -17165.0429688, 12611.3037109, -18828.3105469, 14064.7089844, -31229.7519531, 31439.6074219
1: -13897.1591797, 12216.7861328, -15242.4082031, 13612.1025391, -27509.2617188, 27459.1914062
2: -20281.4785156, 13341.2685547, -22282.5332031, 14852.5244141, -35134.0039062, 35623.8007812
3: -6977.6308594, 19078.0195312, -7717.3305664, 21036.9863281, -28014.6171875, 26795.3496094
4: -22278.3281250, 13183.6787109, -24483.2949219, 14674.1562500, -36952.4804688, 37666.9726562

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8433286, upper bound: 19342.7119489
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8433286, upper bound: 19342.7119489
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -17199.3613281, 12636.5185547, -19032.4433594, 14145.0703125, -31344.4296875, 31668.9609375
1: -13924.8984375, 12241.0957031, -15409.0341797, 13694.4580078, -27619.3554688, 27650.1289062
2: -20321.4941406, 13367.9326172, -22535.5917969, 14936.9892578, -35258.4804688, 35903.5234375
3: -6991.9951172, 19115.5136719, -7770.0375977, 21237.5332031, -28229.5273438, 26885.5488281
4: -22322.4218750, 13210.2080078, -24753.8359375, 14744.1230469, -37066.5351562, 37964.0429688

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7081476, upper bound: 19342.7088192
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7081476, upper bound: 19342.7088192
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -17165.0429688, 12611.3037109, -21017.5312500, 15604.7861328, -32769.8281250, 33628.8359375
1: -13897.1591797, 12216.7861328, -17018.0898438, 15105.8408203, -29002.9941406, 29234.8750000
2: -20281.4785156, 13341.2685547, -24879.6425781, 16479.4179688, -36760.8945312, 38220.9101562
3: -6977.6308594, 19078.0195312, -8578.1494141, 23459.2636719, -30436.8945312, 27656.1679688
4: -22278.3281250, 13183.6787109, -27330.6367188, 16277.9619141, -38556.2773438, 40514.3164062

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7081476, upper bound: 19342.7088192
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7081476, upper bound: 19342.7088192
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -16800.3320312, 12569.3457031, -14973.5019531, 11070.9960938, -27871.3222656, 27542.8437500
1: -13598.5332031, 12167.5263672, -12120.8447266, 10723.0634766, -24321.5976562, 24288.3710938
2: -19886.5273438, 13273.5097656, -17680.5507812, 11712.9384766, -31599.4648438, 30954.0605469
3: -6890.1635742, 18764.3359375, -6115.4731445, 16651.1816406, -23541.3417969, 24879.8046875
4: -21849.7050781, 13103.6064453, -19426.6660156, 11577.5195312, -33427.2226562, 32530.2714844

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_B1_A1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9364883, upper bound: 19342.9358136
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_A1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9364883, upper bound: 19342.9358136
time: 1.06 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -18828.3105469, 14064.7089844, -14934.1025391, 11041.8251953, -29870.1328125, 28998.8125000
1: -15242.4082031, 13612.1025391, -12088.9697266, 10694.9960938, -25937.4042969, 25701.0722656
2: -22282.5332031, 14852.5244141, -17634.5859375, 11682.4072266, -33964.9375000, 32487.1074219
3: -7717.3305664, 21036.9863281, -6099.0957031, 16607.8964844, -24325.2265625, 27136.0820312
4: -24483.2949219, 14674.1562500, -19376.0117188, 11547.1044922, -36030.3984375, 34050.1679688

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_B1_A1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9364883, upper bound: 19342.9358136
time: 0.97 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_A1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9364883, upper bound: 19342.9358136
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -19113.0937500, 14205.7099609, -14857.6435547, 10986.6835938, -30099.7773438, 29063.3535156
1: -15473.9306641, 13753.4726562, -12026.8066406, 10641.2021484, -26115.1328125, 25780.2792969
2: -22631.7929688, 15001.4619141, -17544.5175781, 11624.1933594, -34255.9843750, 32545.9765625
3: -7804.1357422, 21328.9472656, -6069.5864258, 16522.9160156, -24327.0507812, 27398.5332031
4: -24859.4785156, 14807.2119141, -19277.2167969, 11489.2861328, -36348.7656250, 34084.4257812

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8047188, upper bound: 19342.7084733
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7928918, upper bound: 19342.6926236
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -18980.8945312, 14112.5683594, -15604.7080078, 11584.0917969, -30564.9863281, 29717.2773438
1: -15367.6513672, 13664.3476562, -12633.4902344, 11221.5693359, -26589.2207031, 26297.8339844
2: -22478.7167969, 14904.7080078, -18432.1875000, 12250.5087891, -34729.2226562, 33336.8945312
3: -7751.0849609, 21184.7324219, -6392.2875977, 17400.9101562, -25151.9941406, 27577.0195312
4: -24688.8906250, 14710.4316406, -20245.6445312, 12099.2636719, -36788.1484375, 34956.0703125

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7971368, upper bound: 19342.7084733
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7911924, upper bound: 19342.6926435
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -16800.3320312, 12569.3457031, -17199.3613281, 12636.5185547, -29436.8496094, 29768.7070312
1: -13598.5332031, 12167.5263672, -13924.8984375, 12241.0957031, -25839.6289062, 26092.4238281
2: -19886.5273438, 13273.5097656, -20321.4941406, 13367.9326172, -33254.4609375, 33594.9960938
3: -6890.1635742, 18764.3359375, -6991.9951172, 19115.5136719, -26005.6738281, 25756.3300781
4: -21849.7050781, 13103.6064453, -22322.4218750, 13210.2080078, -35059.9101562, 35426.0273438

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_B2_A1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7119489, upper bound: 19342.8433286
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_A1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7119489, upper bound: 19342.8433286
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -18828.3105469, 14064.7089844, -17165.0429688, 12611.3037109, -31439.6074219, 31229.7519531
1: -15242.4082031, 13612.1025391, -13897.1591797, 12216.7861328, -27459.1914062, 27509.2617188
2: -22282.5332031, 14852.5244141, -20281.4785156, 13341.2685547, -35623.8007812, 35134.0000000
3: -7717.3305664, 21036.9863281, -6977.6308594, 19078.0195312, -26795.3496094, 28014.6171875
4: -24483.2949219, 14674.1562500, -22278.3281250, 13183.6787109, -37666.9726562, 36952.4765625

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7119489, upper bound: 19342.8433286
time: 1.11 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7119489, upper bound: 19342.8433286
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -19032.4433594, 14145.0703125, -17199.3613281, 12636.5185547, -31668.9609375, 31344.4316406
1: -15409.0341797, 13694.4580078, -13924.8984375, 12241.0957031, -27650.1289062, 27619.3554688
2: -22535.5917969, 14936.9892578, -20321.4941406, 13367.9326172, -35903.5234375, 35258.4765625
3: -7770.0375977, 21237.5332031, -6991.9951172, 19115.5136719, -26885.5468750, 28229.5273438
4: -24753.8359375, 14744.1230469, -22322.4218750, 13210.2080078, -37964.0429688, 37066.5390625

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7088192, upper bound: 19342.7081476
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7088192, upper bound: 19342.7081476
time: 1.19 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -21017.5312500, 15604.7861328, -17165.0429688, 12611.3037109, -33628.8359375, 32769.8281250
1: -17018.0898438, 15105.8408203, -13897.1591797, 12216.7861328, -29234.8750000, 29002.9960938
2: -24879.6425781, 16479.4179688, -20281.4785156, 13341.2685547, -38220.9101562, 36760.8945312
3: -8578.1494141, 23459.2636719, -6977.6308594, 19078.0195312, -27656.1679688, 30436.8945312
4: -27330.6367188, 16277.9619141, -22278.3281250, 13183.6787109, -40514.3164062, 38556.2773438

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7088192, upper bound: 19342.7081476
time: 1.09 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7088192, upper bound: 19342.7081476
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -16800.3320312, 12569.3457031, -16922.2734375, 12660.8242188, -29461.1562500, 29491.6191406
1: -13598.5332031, 12167.5263672, -13697.1318359, 12255.9824219, -25854.5156250, 25864.6582031
2: -19886.5273438, 13273.5097656, -20030.7636719, 13370.2285156, -33256.7500000, 33304.2734375
3: -6890.1635742, 18764.3359375, -6940.6250000, 18901.0898438, -25791.2539062, 25704.9589844
4: -21849.7050781, 13103.6064453, -22008.0683594, 13198.8886719, -35048.5898438, 35111.6679688

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B2_B1_A1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9378077, upper bound: 19342.9367926
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_A1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9378077, upper bound: 19342.9367926
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -18828.3105469, 14064.7089844, -16891.9980469, 12638.0351562, -31466.3457031, 30956.7070312
1: -15242.4082031, 13612.1025391, -13672.6982422, 12234.3300781, -27476.7382812, 27284.8007812
2: -22282.5332031, 14852.5244141, -19995.7343750, 13346.5195312, -35629.0546875, 34848.2539062
3: -7717.3305664, 21036.9863281, -6927.8613281, 18867.8125000, -26585.1425781, 27964.8476562
4: -24483.2949219, 14674.1562500, -21969.3125000, 13175.0927734, -37658.3867188, 36643.4570312

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B2_B1_A1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9378077, upper bound: 19342.9367926
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_A1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9378077, upper bound: 19342.9367926
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -19151.7734375, 14234.5820312, -16800.3320312, 12569.3457031, -31721.1152344, 31034.9121094
1: -15505.5595703, 13780.9765625, -13598.5332031, 12167.5263672, -27673.0859375, 27379.5097656
2: -22676.7753906, 15031.5585938, -19886.5273438, 13273.5097656, -35950.2851562, 34918.0820312
3: -7819.2700195, 21371.1406250, -6890.1635742, 18764.3359375, -26583.6054688, 28261.3027344
4: -24908.8359375, 14837.4707031, -21849.7050781, 13103.6064453, -38012.4414062, 36687.1757812

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8428644, upper bound: 19342.7107537
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8428644, upper bound: 19342.7107537
time: 0.86 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.82 + 418.44 = 421.26 seconds
