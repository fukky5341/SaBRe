## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_3.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 3613.31311749156


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031)
1: (-2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305)
2: (-2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000)
3: (-1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141)
4: (-3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.62 + 1.94 = 2.55 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -3614.7590211, upper bound: 3614.7590211

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.0483573, upper bound: 3614.4694806
time: 0.61 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7406657, upper bound: 3614.7406664
time: 1.01 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.69 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.69
Output dim: 0, lower bound: -3613.0483573, upper bound: 3614.4694806
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.69
Output dim: 0, lower bound: -3614.7406657, upper bound: 3614.7406664

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -2172.7478027, 1891.9267578, -2457.9965820, 2115.9060059, -4288.6533203, 4349.9233398
1: -1744.2438965, 1855.7540283, -1979.1912842, 2073.4768066, -3817.7207031, 3834.9450684
2: -2557.1499023, 2009.1342773, -2920.3452148, 2248.3352051, -4805.4853516, 4929.4794922
3: -990.0590820, 2595.9711914, -1108.4735107, 2930.8200684, -3920.8789062, 3704.4448242
4: -2815.3046875, 1956.6667480, -3213.3332520, 2190.3139648, -5005.6181641, 5169.9995117

Time for backsubstitution: 0.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0461844, upper bound: 3613.0461844
time: 0.86 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.0461844, upper bound: 3614.4694806
time: 0.57 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -2500.9851074, 2149.7963867, -2502.3757324, 2150.9697266, -4651.9545898, 4652.1718750
1: -2014.5847168, 2106.3789062, -2015.7012939, 2107.5605469, -4122.1455078, 4122.0800781
2: -2976.4731445, 2283.9982910, -2978.1022949, 2285.2739258, -5261.7460938, 5262.0996094
3: -1125.7036133, 2982.2333984, -1126.3112793, 2983.9003906, -4109.6040039, 4108.5449219
4: -3274.6416016, 2225.7395020, -3276.4265137, 2226.9489746, -5501.5903320, 5502.1655273

Time for backsubstitution: 0.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.4694806, upper bound: 3613.0483573
time: 0.72 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.4694806, upper bound: 3614.7406664
time: 0.71 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.04 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.04
Output dim: 0, lower bound: -3613.0461844, upper bound: 3613.0461844
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.04
Output dim: 0, lower bound: -3613.0461844, upper bound: 3614.4694806
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.04
Output dim: 0, lower bound: -3614.4694806, upper bound: 3613.0483573
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.04
Output dim: 0, lower bound: -3614.4694806, upper bound: 3614.7406664

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -2172.7478027, 1891.9267578, -2485.0832520, 2137.5415039, -4310.2890625, 4377.0097656
1: -1744.2438965, 1855.7540283, -2001.3973389, 2094.5544434, -3838.7983398, 3857.1513672
2: -2557.1499023, 2009.1342773, -2955.1914062, 2271.1857910, -4828.3359375, 4964.3251953
3: -990.0590820, 2595.9711914, -1119.5476074, 2963.0737305, -3953.1328125, 3715.5187988
4: -2815.3046875, 1956.6667480, -3251.5161133, 2212.8542480, -5028.1577148, 5208.1826172

Time for backsubstitution: 0.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9434311, upper bound: 3612.9078361
time: 0.63 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9434265, upper bound: 3614.4058487
time: 0.76 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -2485.0832520, 2137.5415039, -2172.7478027, 1891.9267578, -4377.0097656, 4310.2890625
1: -2001.3973389, 2094.5544434, -1744.2438965, 1855.7540283, -3857.1513672, 3838.7983398
2: -2955.1914062, 2271.1857910, -2557.1499023, 2009.1342773, -4964.3247070, 4828.3359375
3: -1119.5476074, 2963.0737305, -990.0590820, 2595.9711914, -3715.5187988, 3953.1323242
4: -3251.5161133, 2212.8542480, -2815.3046875, 1956.6667480, -5208.1826172, 5028.1577148

Time for backsubstitution: 0.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3639448, upper bound: 3612.9455494
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.4058443, upper bound: 3612.9456832
time: 1.02 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -2500.9851074, 2149.7963867, -2500.9851074, 2149.7963867, -4650.7812500, 4650.7812500
1: -2014.5847168, 2106.3789062, -2014.5847168, 2106.3789062, -4120.9638672, 4120.9638672
2: -2976.4731445, 2283.9982910, -2976.4731445, 2283.9982910, -5260.4716797, 5260.4711914
3: -1125.7036133, 2982.2333984, -1125.7036133, 2982.2333984, -4107.9370117, 4107.9370117
4: -3274.6416016, 2225.7395020, -3274.6416016, 2225.7395020, -5500.3803711, 5500.3803711

Time for backsubstitution: 0.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.9571789, upper bound: 3613.6736965
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6715083, upper bound: 3613.6736835
time: 0.71 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.08 seconds
NS_A1_B2_B1, status: Status.VERIFIED, split count: 3, time: 2.08
Output dim: 0, lower bound: -3612.9434311, upper bound: 3612.9078361
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.08
Output dim: 0, lower bound: -3612.9434265, upper bound: 3614.4058487
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.08
Output dim: 0, lower bound: -3614.3639448, upper bound: 3612.9455494
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.08
Output dim: 0, lower bound: -3614.4058443, upper bound: 3612.9456832
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.08
Output dim: 0, lower bound: -3613.9571789, upper bound: 3613.6736965
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.08
Output dim: 0, lower bound: -3613.6715083, upper bound: 3613.6736835

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -2172.7478027, 1891.9267578, -2468.1118164, 2122.5776367, -4295.3251953, 4360.0385742
1: -1744.2438965, 1855.7540283, -1987.7734375, 2079.7336426, -3823.9772949, 3843.5273438
2: -2557.1499023, 2009.1342773, -2935.1994629, 2255.2207031, -4812.3706055, 4944.3325195
3: -990.0590820, 2595.9711914, -1111.7760010, 2942.7993164, -3932.8583984, 3707.7470703
4: -2815.3046875, 1956.6667480, -3229.4973145, 2197.3435059, -5012.6459961, 5186.1640625

Time for backsubstitution: 0.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_B2_B1

### Relational analysis result of NS_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.7424471, upper bound: 3613.5146677
time: 0.60 seconds

## Relational analysis of NS_A1_B2_B2_B2

### Relational analysis result of NS_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9456755, upper bound: 3614.4058420
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2386.3383789, 2048.7043457, -2170.0800781, 1889.5325928, -4275.8706055, 4218.7836914
1: -1921.7612305, 2006.3262939, -1742.1030273, 1853.3813477, -3775.1425781, 3748.4291992
2: -2836.2751465, 2176.7814941, -2553.9772949, 2006.5783691, -4842.8535156, 4730.7587891
3: -1073.9309082, 2840.8671875, -988.8171387, 2592.7150879, -3666.6450195, 3829.6843262
4: -3120.7587891, 2120.8178711, -2811.8142090, 1954.1976318, -5074.9565430, 4932.6318359

Time for backsubstitution: 0.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.8242755, upper bound: 3612.5725734
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2655931, upper bound: 3612.9453762
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A1_A2

### Relational analysis result of NS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3638988, upper bound: 3612.9453759
time: 0.61 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2468.1118164, 2122.5776367, -2172.7478027, 1891.9267578, -4360.0385742, 4295.3251953
1: -1987.7734375, 2079.7336426, -1744.2438965, 1855.7540283, -3843.5273438, 3823.9772949
2: -2935.1994629, 2255.2207031, -2557.1499023, 2009.1342773, -4944.3330078, 4812.3706055
3: -1111.7760010, 2942.7993164, -990.0590820, 2595.9711914, -3707.7470703, 3932.8583984
4: -3229.4975586, 2197.3435059, -2815.3046875, 1956.6667480, -5186.1640625, 5012.6459961

Time for backsubstitution: 0.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5146677, upper bound: 3612.7424471
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.4058377, upper bound: 3612.9456755
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2271.4199219, 1964.2563477, -2500.9851074, 2149.7963867, -4421.2163086, 4465.2407227
1: -1830.2175293, 1926.7331543, -2014.5847168, 2106.3789062, -3936.5964355, 3941.3178711
2: -2710.7407227, 2087.3911133, -2976.4731445, 2283.9982910, -4994.7392578, 5063.8627930
3: -1026.1483154, 2722.9445801, -1125.7036133, 2982.2333984, -4008.3818359, 3848.6481934
4: -2980.7939453, 2034.3486328, -3274.6416016, 2225.7395020, -5206.5322266, 5308.9902344

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6632115, upper bound: 3613.6631935
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6632115, upper bound: 3613.6736832
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4050.1276855, 3653.4504395, -2467.3413086, 2125.5085449, -6175.6342773, 5948.4262695
1: -3265.4714355, 3596.7358398, -1988.0856934, 2082.7053223, -5348.1752930, 5441.3217773
2: -4926.2226562, 3866.6953125, -2938.5334473, 2258.3037109, -7155.9638672, 6645.8266602
3: -1813.9953613, 4968.0659180, -1112.7136230, 2947.0034180, -4719.2490234, 6026.6279297
4: -5388.5708008, 3756.5537109, -3232.3481445, 2200.7177734, -7572.2597656, 6821.2832031

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0523785, upper bound: 3612.9865107
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936278, upper bound: 3613.3946385
time: 8.86 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 10.24 seconds
NS_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 10.24
Output dim: 0, lower bound: -3612.7424471, upper bound: 3613.5146677
NS_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 10.24
Output dim: 0, lower bound: -3612.9456755, upper bound: 3614.4058420
NS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 10.24
Output dim: 0, lower bound: -3614.2655931, upper bound: 3612.9453762
NS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 10.24
Output dim: 0, lower bound: -3614.3638988, upper bound: 3612.9453759
NS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 10.24
Output dim: 0, lower bound: -3613.5146677, upper bound: 3612.7424471
NS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 10.24
Output dim: 0, lower bound: -3614.4058377, upper bound: 3612.9456755
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 10.24
Output dim: 0, lower bound: -3613.6632115, upper bound: 3613.6631935
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 10.24
Output dim: 0, lower bound: -3613.6632115, upper bound: 3613.6736832
NS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 10.24
Output dim: 0, lower bound: -3613.0523785, upper bound: 3612.9865107
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 10.24
Output dim: 0, lower bound: -3613.3936278, upper bound: 3613.3946385

## BFS NS instance: NS_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -2158.6706543, 1879.8715820, -2557.0830078, 2211.9174805, -4370.5878906, 4436.9545898
1: -1732.9484863, 1843.8437500, -2059.3234863, 2166.3549805, -3899.3034668, 3903.1672363
2: -2540.5273438, 1996.1988525, -3041.0173340, 2348.6628418, -4889.1904297, 5037.2153320
3: -983.6867676, 2579.2321777, -1153.8485107, 3053.1987305, -4036.8854980, 3733.0803223
4: -2797.0300293, 1944.1992188, -3345.9060059, 2291.5244141, -5088.5541992, 5290.1049805

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_B2_B1_B1

### Relational analysis result of NS_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.7424471, upper bound: 3613.4737078
time: 0.67 seconds

## Relational analysis of NS_A1_B2_B2_B1_B2

### Relational analysis result of NS_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.7422297, upper bound: 3613.5146677
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -2172.7478027, 1891.9267578, -2461.3723145, 2116.8151855, -4289.5629883, 4353.2988281
1: -1744.2438965, 1855.7540283, -1982.3831787, 2074.0979004, -3818.3415527, 3838.1372070
2: -2557.1499023, 2009.1342773, -2927.4079590, 2249.0961914, -4806.2460938, 4936.5419922
3: -990.0590820, 2595.9711914, -1108.7523193, 2934.9311523, -3924.9899902, 3704.7236328
4: -2815.3046875, 1956.6667480, -3220.8557129, 2191.4238281, -5006.7285156, 5177.5224609

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.2687023, upper bound: 3614.0224459
time: 0.77 seconds

## Relational analysis of NS_A1_B2_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.5957106, upper bound: 3614.0224557
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -2140.4040527, 1834.1047363, -2095.8046875, 1826.9348145, -3967.3388672, 3929.9091797
1: -1723.7551270, 1794.4382324, -1682.7038574, 1792.4254150, -3516.1806641, 3477.1420898
2: -2547.1953125, 1948.1474609, -2467.2934570, 1940.8488770, -4488.0439453, 4415.4409180
3: -960.8016968, 2542.0007324, -955.3113403, 2504.8374023, -3465.6386719, 3497.3120117
4: -2801.3291016, 1898.2889404, -2715.8264160, 1890.0585938, -4691.3876953, 4614.1147461

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_A1_B1

### Relational analysis result of NS_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0073006, upper bound: 3612.2626554
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2

### Relational analysis result of NS_A2_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.6037468, upper bound: 3612.5955579
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -2344.2265625, 2012.3312988, -2149.3027344, 1871.9624023, -4216.1889648, 4161.6337891
1: -1888.0964355, 1970.5882568, -1725.4643555, 1836.1021729, -3724.1982422, 3696.0522461
2: -2787.2050781, 2137.7880859, -2529.5754395, 1987.8710938, -4775.0756836, 4667.3632812
3: -1054.3319092, 2791.2412109, -979.2920532, 2568.1818848, -3622.5136719, 3770.5329590
4: -3066.7282715, 2083.2150879, -2784.9982910, 1936.0859375, -5002.8129883, 4868.2133789

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_A2_B1

### Relational analysis result of NS_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3630421, upper bound: 3612.7873991
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2

### Relational analysis result of NS_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3630421, upper bound: 3612.9453759
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -2557.0830078, 2211.9174805, -2158.6706543, 1879.8715820, -4436.9545898, 4370.5878906
1: -2059.3234863, 2166.3549805, -1732.9484863, 1843.8437500, -3903.1672363, 3899.3034668
2: -3041.0173340, 2348.6628418, -2540.5273438, 1996.1988525, -5037.2148438, 4889.1904297
3: -1153.8485107, 3053.1987305, -983.6867676, 2579.2321777, -3733.0803223, 4036.8854980
4: -3345.9060059, 2291.5244141, -2797.0300293, 1944.1992188, -5290.1049805, 5088.5541992

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_A1_A1

### Relational analysis result of NS_A2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4737078, upper bound: 3612.7424471
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_A1_A2

### Relational analysis result of NS_A2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5146677, upper bound: 3612.7422297
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -2461.3723145, 2116.8151855, -2172.7478027, 1891.9267578, -4353.2988281, 4289.5629883
1: -1982.3831787, 2074.0979004, -1744.2438965, 1855.7540283, -3838.1372070, 3818.3415527
2: -2927.4079590, 2249.0961914, -2557.1499023, 2009.1342773, -4936.5419922, 4806.2460938
3: -1108.7523193, 2934.9311523, -990.0590820, 2595.9711914, -3704.7236328, 3924.9899902
4: -3220.8557129, 2191.4238281, -2815.3046875, 1956.6667480, -5177.5224609, 5006.7285156

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_A2_B1

### Relational analysis result of NS_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0224459, upper bound: 3612.2687023
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2

### Relational analysis result of NS_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0224510, upper bound: 3612.5957106
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2271.4199219, 1964.2563477, -2271.4199219, 1964.2563477, -4235.6757812, 4235.6757812
1: -1830.2175293, 1926.7331543, -1830.2175293, 1926.7331543, -3756.9506836, 3756.9506836
2: -2710.7407227, 2087.3911133, -2710.7407227, 2087.3911133, -4798.1318359, 4798.1318359
3: -1026.1483154, 2722.9445801, -1026.1483154, 2722.9445801, -3749.0927734, 3749.0927734
4: -2980.7939453, 2034.3486328, -2980.7939453, 2034.3486328, -5015.1425781, 5015.1425781

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2271.4199219, 1964.2563477, -4045.9091797, 3648.9858398, -5750.7431641, 6010.1650391
1: -1830.2175293, 1926.7331543, -3262.1235352, 3592.2441406, -5281.2597656, 5188.8559570
2: -2710.7407227, 2087.3911133, -4921.1171875, 3861.9504395, -6417.8452148, 6986.0214844
3: -1026.1483154, 2722.9445801, -1811.9459229, 4962.5239258, -5936.8120117, 4495.8408203
4: -2980.7939453, 2034.3486328, -5382.9443359, 3751.9707031, -6569.7900391, 7406.4018555

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4080318, upper bound: 3613.3946436
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9209324, upper bound: 3613.3946436
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4047.4299316, 3652.8283691, -4045.2041016, 3623.2385254, -7447.8291016, 7460.7583008
1: -3263.3154297, 3596.4865723, -3262.7951660, 3565.3437500, -6636.5913086, 6657.1821289
2: -4924.0244141, 3866.2978516, -4908.1582031, 3834.6760254, -8481.0761719, 8494.8037109
3: -1812.9794922, 4966.1181641, -1806.4891357, 4932.6279297, -6613.9580078, 6621.2319336
4: -5386.0039062, 3756.0119629, -5365.0288086, 3726.1301270, -8834.6845703, 8839.8193359

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9864691, upper bound: 3613.3946382
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9864691, upper bound: 3613.3946406
time: 0.76 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.15 seconds
NS_A1_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -3612.7424471, upper bound: 3613.4737078
NS_A1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -3612.7422297, upper bound: 3613.5146677
NS_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -3612.2687023, upper bound: 3614.0224459
NS_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -3612.5957106, upper bound: 3614.0224557
NS_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -3614.0073006, upper bound: 3612.2626554
NS_A2_B1_A1_A1_B2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -3612.6037468, upper bound: 3612.5955579
NS_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -3614.3630421, upper bound: 3612.7873991
NS_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -3614.3630421, upper bound: 3612.9453759
NS_A2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -3613.4737078, upper bound: 3612.7424471
NS_A2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -3613.5146677, upper bound: 3612.7422297
NS_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -3614.0224459, upper bound: 3612.2687023
NS_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -3614.0224510, upper bound: 3612.5957106
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -3613.4080318, upper bound: 3613.3946436
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -3612.9209324, upper bound: 3613.3946436
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -3612.9864691, upper bound: 3613.3946382
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.15
Output dim: 0, lower bound: -3612.9864691, upper bound: 3613.3946406

## BFS NS instance: NS_A1_B2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -2158.6706543, 1879.8715820, -2478.1774902, 2144.8903809, -4303.5610352, 4358.0488281
1: -1732.9484863, 1843.8437500, -1996.1049805, 2101.0346680, -3833.9831543, 3839.9484863
2: -2540.5273438, 1996.1988525, -2949.6979980, 2277.8549805, -4818.3823242, 4945.8959961
3: -983.6867676, 2579.2321777, -1118.4680176, 2960.8491211, -3944.5356445, 3697.6997070
4: -2797.0300293, 1944.1992188, -3244.9968262, 2222.6489258, -5019.6787109, 5189.1958008

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_B2_B1_B1_A1

### Relational analysis result of NS_A1_B2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.7702119, upper bound: 3613.4735997
time: 0.66 seconds

## Relational analysis of NS_A1_B2_B2_B1_B1_A2

### Relational analysis result of NS_A1_B2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.7702119, upper bound: 3613.4737141
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -2147.6245117, 1870.6962891, -2494.9741211, 2154.3530273, -4301.9765625, 4365.6704102
1: -1724.1331787, 1834.8715820, -2010.4708252, 2110.0100098, -3834.1430664, 3845.3422852
2: -2527.7448730, 1986.5054932, -2970.7702637, 2287.4985352, -4815.2431641, 4957.2749023
3: -978.8170166, 2566.3854980, -1125.6015625, 2981.4743652, -3960.2907715, 3691.9868164
4: -2782.8669434, 1934.7774658, -3267.1845703, 2231.9599609, -5014.8271484, 5201.9619141

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_B2_B1_B2_A1

### Relational analysis result of NS_A1_B2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.7704042, upper bound: 3613.5145514
time: 0.77 seconds

## Relational analysis of NS_A1_B2_B2_B1_B2_A2

### Relational analysis result of NS_A1_B2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.7704042, upper bound: 3613.5146734
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -1998.9207764, 1737.3414307, -2367.8374023, 2035.6910400, -4034.6118164, 4105.1787109
1: -1604.2591553, 1703.2763672, -1906.5493164, 1994.1721191, -3598.4311523, 3609.8256836
2: -2352.9548340, 1845.4056396, -2816.3610840, 2163.2529297, -4516.2075195, 4661.7661133
3: -910.6644897, 2386.7770996, -1066.9450684, 2822.4333496, -3733.0974121, 3453.7221680
4: -2591.2009277, 1797.2329102, -3099.1982422, 2106.9916992, -4698.1918945, 4896.4311523

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B2_B2_A1_B1

### Relational analysis result of NS_A1_B2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8928309, upper bound: 3614.0170694
time: 0.75 seconds

## Relational analysis of NS_A1_B2_B2_B2_A1_B2

### Relational analysis result of NS_A1_B2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.9409444, upper bound: 3613.9245999
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -2136.2788086, 1860.7567139, -2444.6774902, 2102.5998535, -4238.8779297, 4305.4335938
1: -1714.7510986, 1825.5142822, -1969.0103760, 2060.1835938, -3774.9345703, 3794.5246582
2: -2514.0349121, 1976.5239258, -2908.0451660, 2234.1186523, -4748.1528320, 4884.5693359
3: -973.8831177, 2553.4851074, -1101.3217773, 2915.5563965, -3889.4392090, 3654.8068848
4: -2768.0297852, 1924.3913574, -3199.5629883, 2176.6733398, -4944.7031250, 5123.9541016

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B2_B2_A2_B1

### Relational analysis result of NS_A1_B2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8425709, upper bound: 3614.0170751
time: 0.67 seconds

## Relational analysis of NS_A1_B2_B2_B2_A2_B2

### Relational analysis result of NS_A1_B2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8425709, upper bound: 3613.9245073
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -2075.9670410, 1777.7312012, -1938.7263184, 1687.0545654, -3763.0214844, 3716.4575195
1: -1671.8638916, 1738.7159424, -1556.1984863, 1654.1333008, -3325.9970703, 3294.9143066
2: -2470.7639160, 1888.0756836, -2282.6088867, 1792.3089600, -4263.0727539, 4170.6845703
3: -931.3259888, 2464.4819336, -883.6118164, 2315.9826660, -3247.3085938, 3348.0937500
4: -2717.6254883, 1839.7073975, -2513.4553223, 1745.4670410, -4463.0927734, 4353.1625977

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9372101, upper bound: 3612.0740095
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.9008886, upper bound: 3612.0913295
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -2344.1630859, 2012.2916260, -1935.8311768, 1681.4370117, -4025.5996094, 3948.1223145
1: -1888.0466309, 1970.5491943, -1553.4782715, 1648.7939453, -3536.8405762, 3524.0270996
2: -2787.1389160, 2137.7470703, -2277.6936035, 1786.7449951, -4573.8837891, 4415.4404297
3: -1054.3088379, 2791.1757812, -880.5640869, 2306.0104980, -3360.3193359, 3671.7397461
4: -3066.6542969, 2083.1748047, -2506.8803711, 1738.3172607, -4804.9711914, 4590.0551758

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_A2_B1_B1

### Relational analysis result of NS_A2_B1_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.5702959, upper bound: 3612.1833855
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A1_A2_B1_B2

### Relational analysis result of NS_A2_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0217443, upper bound: 3612.5955334
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -2344.1315918, 2012.2584229, -2127.1401367, 1853.0717773, -4197.2031250, 4139.3979492
1: -1888.0181885, 1970.5177002, -1707.7254639, 1817.5609131, -3705.5791016, 3678.2431641
2: -2787.0812988, 2137.7114258, -2503.5363770, 1967.7625732, -4754.8437500, 4641.2475586
3: -1054.2947998, 2791.1276855, -969.1143188, 2541.9597168, -3596.2543945, 3760.2414551
4: -3066.5930176, 2083.1381836, -2756.3745117, 1916.6109619, -4983.2041016, 4839.5126953

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_A2_B2_B1

### Relational analysis result of NS_A2_B1_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.5702959, upper bound: 3612.2627279
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2_B2

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0217443, upper bound: 3612.5955334
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A1

### Backsubstitution after applying NS history:
0: -2478.1774902, 2144.8906250, -2158.6706543, 1879.8715820, -4358.0488281, 4303.5615234
1: -1996.1049805, 2101.0349121, -1732.9484863, 1843.8437500, -3839.9484863, 3833.9831543
2: -2949.6979980, 2277.8549805, -2540.5273438, 1996.1988525, -4945.8964844, 4818.3823242
3: -1118.4680176, 2960.8491211, -983.6867676, 2579.2321777, -3697.6997070, 3944.5356445
4: -3244.9968262, 2222.6489258, -2797.0300293, 1944.1992188, -5189.1953125, 5019.6787109

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_A1_A1_B1

### Relational analysis result of NS_A2_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4735997, upper bound: 3611.7702119
time: 3.02 seconds

## Relational analysis of NS_A2_B1_A2_A1_A1_B2

### Relational analysis result of NS_A2_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4735997, upper bound: 3612.7422297
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A2

### Backsubstitution after applying NS history:
0: -2494.9741211, 2154.3527832, -2147.6245117, 1870.6962891, -4365.6704102, 4301.9765625
1: -2010.4707031, 2110.0095215, -1724.1331787, 1834.8715820, -3845.3422852, 3834.1425781
2: -2970.7702637, 2287.4982910, -2527.7448730, 1986.5054932, -4957.2749023, 4815.2431641
3: -1125.6015625, 2981.4741211, -978.8170166, 2566.3854980, -3691.9868164, 3960.2905273
4: -3267.1850586, 2231.9599609, -2782.8669434, 1934.7774658, -5201.9619141, 5014.8271484

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_A1_A2_B1

### Relational analysis result of NS_A2_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5145458, upper bound: 3611.7704042
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A2_A1_A2_B2

### Relational analysis result of NS_A2_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5145458, upper bound: 3612.7422297
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -2367.8374023, 2035.6910400, -1998.9207764, 1737.3414307, -4105.1787109, 4034.6118164
1: -1906.5493164, 1994.1721191, -1604.2591553, 1703.2763672, -3609.8256836, 3598.4311523
2: -2816.3610840, 2163.2529297, -2352.9548340, 1845.4056396, -4661.7656250, 4516.2075195
3: -1066.9450684, 2822.4333496, -910.6644897, 2386.7770996, -3453.7221680, 3733.0974121
4: -3099.1982422, 2106.9916992, -2591.2009277, 1797.2329102, -4896.4306641, 4698.1923828

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0170694, upper bound: 3611.8928309
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.9245938, upper bound: 3611.9409444
time: 1.00 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -2444.6774902, 2102.5998535, -2136.2788086, 1860.7567139, -4305.4331055, 4238.8779297
1: -1969.0103760, 2060.1835938, -1714.7510986, 1825.5142822, -3794.5246582, 3774.9345703
2: -2908.0451660, 2234.1186523, -2514.0349121, 1976.5239258, -4884.5693359, 4748.1533203
3: -1101.3217773, 2915.5563965, -973.8831177, 2553.4851074, -3654.8068848, 3889.4392090
4: -3199.5629883, 2176.6733398, -2768.0297852, 1924.3913574, -5123.9541016, 4944.7031250

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0170701, upper bound: 3611.8425709
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.9245012, upper bound: 3611.8425709
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2032.5021973, 1767.8198242, -4041.7773438, 3644.6513672, -5511.4052734, 5809.5971680
1: -1637.2110596, 1736.3991699, -3258.8447266, 3587.8864746, -5087.1230469, 4995.2441406
2: -2434.3435059, 1878.1309814, -4916.1054688, 3857.3552246, -6143.0214844, 6778.3247070
3: -919.7875977, 2453.9553223, -1809.9702148, 4957.1191406, -5828.0874023, 4228.4921875
4: -2675.7617188, 1831.0811768, -5377.4233398, 3747.5317383, -6266.4326172, 7204.4345703

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B2_A1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2078406, upper bound: 3613.2902244
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3780854, upper bound: 3613.2903514
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3858.8288574, 3476.9750977, -4047.4299316, 3652.8283691, -7264.5375977, 7285.9853516
1: -3110.0395508, 3427.8459473, -3263.3154297, 3596.4865723, -6497.1440430, 6484.9438477
2: -4695.0322266, 3685.5327148, -4924.0244141, 3866.2978516, -8267.1416016, 8314.6826172
3: -1728.6866455, 4724.4501953, -1812.9794922, 4966.1181641, -6535.4306641, 6396.1777344
4: -5128.0468750, 3577.4528809, -5386.0039062, 3756.0119629, -8586.5820312, 8669.4492188

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9209280, upper bound: 3612.9865199
time: 0.93 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9209280, upper bound: 3613.3946398
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3842.9118652, 3492.4929199, -4045.2041016, 3623.2385254, -7231.0971680, 7281.4868164
1: -3097.1203613, 3444.2551270, -3262.7951660, 3565.3437500, -6460.8662109, 6487.5634766
2: -4690.9912109, 3699.9401855, -4908.1582031, 3834.6760254, -8229.9580078, 8307.1835938
3: -1723.5593262, 4738.4677734, -1806.4891357, 4932.6279297, -6514.7915039, 6381.7573242
4: -5127.6469727, 3592.6191406, -5365.0288086, 3726.1301270, -8556.1152344, 8656.4521484

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9476928, upper bound: 3612.9209271
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9476928, upper bound: 3613.0334912
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6017.1333008, 5523.3945312, -4045.2041016, 3623.2385254, -9278.1591797, 8939.0478516
1: -4845.6884766, 5456.0732422, -3262.7951660, 3565.3437500, -8102.1298828, 8181.4633789
2: -7362.9492188, 5864.5083008, -4908.1582031, 3834.6760254, -10687.0800781, 10083.0781250
3: -2705.6147461, 7431.8520508, -1806.4891357, 4932.6279297, -7344.4853516, 8915.8857422
4: -8032.2280273, 5679.9628906, -5365.0288086, 3726.1301270, -11243.8603516, 10355.3085938

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9476967, upper bound: 3612.9209307
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9476928, upper bound: 3613.3935470
time: 0.65 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 1.99 seconds
NS_A1_B2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 1.99
Output dim: 0, lower bound: -3611.7702119, upper bound: 3613.4735997
NS_A1_B2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 1.99
Output dim: 0, lower bound: -3611.7702119, upper bound: 3613.4737141
NS_A1_B2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 1.99
Output dim: 0, lower bound: -3611.7704042, upper bound: 3613.5145514
NS_A1_B2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 1.99
Output dim: 0, lower bound: -3611.7704042, upper bound: 3613.5146734
NS_A1_B2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.99
Output dim: 0, lower bound: -3611.8928309, upper bound: 3614.0170694
NS_A1_B2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.99
Output dim: 0, lower bound: -3611.9409444, upper bound: 3613.9245999
NS_A1_B2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.99
Output dim: 0, lower bound: -3611.8425709, upper bound: 3614.0170751
NS_A1_B2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.99
Output dim: 0, lower bound: -3611.8425709, upper bound: 3613.9245073
NS_A2_B1_A1_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 1.99
Output dim: 0, lower bound: -3612.9372101, upper bound: 3612.0740095
NS_A2_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 1.99
Output dim: 0, lower bound: -3613.9008886, upper bound: 3612.0913295
NS_A2_B1_A1_A2_B1_B1, status: Status.VERIFIED, split count: 6, time: 1.99
Output dim: 0, lower bound: -3612.5702959, upper bound: 3612.1833855
NS_A2_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 1.99
Output dim: 0, lower bound: -3614.0217443, upper bound: 3612.5955334
NS_A2_B1_A1_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 1.99
Output dim: 0, lower bound: -3612.5702959, upper bound: 3612.2627279
NS_A2_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 1.99
Output dim: 0, lower bound: -3614.0217443, upper bound: 3612.5955334
NS_A2_B1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.99
Output dim: 0, lower bound: -3613.4735997, upper bound: 3611.7702119
NS_A2_B1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.99
Output dim: 0, lower bound: -3613.4735997, upper bound: 3612.7422297
NS_A2_B1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.99
Output dim: 0, lower bound: -3613.5145458, upper bound: 3611.7704042
NS_A2_B1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.99
Output dim: 0, lower bound: -3613.5145458, upper bound: 3612.7422297
NS_A2_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 1.99
Output dim: 0, lower bound: -3614.0170694, upper bound: 3611.8928309
NS_A2_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 1.99
Output dim: 0, lower bound: -3613.9245938, upper bound: 3611.9409444
NS_A2_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 1.99
Output dim: 0, lower bound: -3614.0170701, upper bound: 3611.8425709
NS_A2_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 1.99
Output dim: 0, lower bound: -3613.9245012, upper bound: 3611.8425709
NS_A2_B2_A1_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 1.99
Output dim: 0, lower bound: -3613.2078406, upper bound: 3613.2902244
NS_A2_B2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 1.99
Output dim: 0, lower bound: -3613.3780854, upper bound: 3613.2903514
NS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 1.99
Output dim: 0, lower bound: -3612.9209280, upper bound: 3612.9865199
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.99
Output dim: 0, lower bound: -3612.9209280, upper bound: 3613.3946398
NS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 1.99
Output dim: 0, lower bound: -3612.9476928, upper bound: 3612.9209271
NS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 1.99
Output dim: 0, lower bound: -3612.9476928, upper bound: 3613.0334912
NS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 1.99
Output dim: 0, lower bound: -3612.9476967, upper bound: 3612.9209307
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.99
Output dim: 0, lower bound: -3612.9476928, upper bound: 3613.3935470

## BFS NS instance: NS_A1_B2_B2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -2082.4411621, 1816.4946289, -2478.1323242, 2144.8574219, -4227.2973633, 4294.6269531
1: -1672.2194824, 1782.3671875, -1996.0679932, 2101.0026855, -3773.2221680, 3778.4350586
2: -2452.7009277, 1929.6461182, -2949.6411133, 2277.8205566, -4730.5214844, 4879.2871094
3: -950.3276367, 2491.3442383, -1118.4505615, 2960.7973633, -3911.1247559, 3609.7944336
4: -2699.4194336, 1879.0957031, -3244.9345703, 2222.6145020, -4922.0341797, 5124.0302734

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B2_B1_B1_A1_B1

### Relational analysis result of NS_A1_B2_B2_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.5828378, upper bound: 3613.0043742
time: 0.65 seconds

## Relational analysis of NS_A1_B2_B2_B1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_B2_B1_B1_A1_A1

### Relational analysis result of NS_A1_B2_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.6775390, upper bound: 3613.4734865
time: 0.72 seconds

## Relational analysis of NS_A1_B2_B2_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_B1_B1_A1_B1

### Relational analysis result of NS_A1_B2_B2_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.7581409, upper bound: 3611.3824795
time: 0.66 seconds

## Relational analysis of NS_A1_B2_B2_B1_B1_A1_B2

### Relational analysis result of NS_A1_B2_B2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.7599242, upper bound: 3613.4736060
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_B2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -2111.0319824, 1836.8913574, -2478.0554199, 2144.8020020, -4255.8339844, 4314.9467773
1: -1694.9891357, 1802.7271729, -1996.0053711, 2100.9492188, -3795.9377441, 3798.7324219
2: -2485.4467773, 1952.0012207, -2949.5437012, 2277.7634277, -4763.2094727, 4901.5449219
3: -962.4727173, 2524.1877441, -1118.4224854, 2960.7084961, -3923.1809082, 3642.6103516
4: -2735.7641602, 1899.9638672, -3244.8283691, 2222.5578613, -4958.3212891, 5144.7915039

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B2_B1_B1_A2_B1

### Relational analysis result of NS_A1_B2_B2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.5828378, upper bound: 3613.4707041
time: 0.71 seconds

## Relational analysis of NS_A1_B2_B2_B1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_B2_B1_B1_A2_A1

### Relational analysis result of NS_A1_B2_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.6775390, upper bound: 3613.4734918
time: 0.70 seconds

## Relational analysis of NS_A1_B2_B2_B1_B1_A2_A2

### Relational analysis result of NS_A1_B2_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.6775390, upper bound: 3613.4734865
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_B2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -2082.4411621, 1816.4946289, -2494.9543457, 2154.3381348, -4236.7773438, 4311.4482422
1: -1672.2194824, 1782.3671875, -2010.4550781, 2109.9956055, -3782.2150879, 3792.8222656
2: -2452.7009277, 1929.6461182, -2970.7456055, 2287.4833984, -4740.1845703, 4900.3916016
3: -950.3276367, 2491.3442383, -1125.5938721, 2981.4509277, -3931.7785645, 3616.9377441
4: -2699.4194336, 1879.0957031, -3267.1584473, 2231.9443359, -4931.3637695, 5146.2539062

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_B2_B1_B2_A1_A1

### Relational analysis result of NS_A1_B2_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.6775390, upper bound: 3613.5144836
time: 0.69 seconds

## Relational analysis of NS_A1_B2_B2_B1_B2_A1_A2

### Relational analysis result of NS_A1_B2_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.6775390, upper bound: 3613.5145514
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_B2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -2111.0319824, 1836.8913574, -2494.8828125, 2154.2866211, -4265.3183594, 4331.7739258
1: -1694.9891357, 1802.7271729, -2010.3967285, 2109.9460449, -3804.9350586, 3813.1240234
2: -2485.4467773, 1952.0012207, -2970.6552734, 2287.4299316, -4772.8759766, 4922.6562500
3: -962.4727173, 2524.1877441, -1125.5673828, 2981.3688965, -3943.8413086, 3649.7551270
4: -2735.7641602, 1899.9638672, -3267.0595703, 2231.8908691, -4967.6552734, 5167.0219727

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_B2_B1_B2_A2_A1

### Relational analysis result of NS_A1_B2_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.6775390, upper bound: 3613.5143700
time: 0.80 seconds

## Relational analysis of NS_A1_B2_B2_B1_B2_A2_A2

### Relational analysis result of NS_A1_B2_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.6775390, upper bound: 3613.5146042
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1987.2487793, 1727.2838135, -2397.9648438, 2057.9523926, -4045.2011719, 4125.2485352
1: -1594.8696289, 1693.3439941, -1930.3010254, 2014.5875244, -3609.4570312, 3623.6447754
2: -2339.1728516, 1834.6950684, -2849.9020996, 2184.5903320, -4523.7631836, 4684.5961914
3: -905.3850098, 2372.7167969, -1077.5959473, 2855.4157715, -3760.8005371, 3450.3127441
4: -2576.0402832, 1786.7614746, -3137.6254883, 2129.1381836, -4705.1787109, 4924.3867188

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.7862918, upper bound: 3613.9233561
time: 0.71 seconds

## Relational analysis of NS_A1_B2_B2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.7862918, upper bound: 3613.9245999
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1972.7789307, 1711.5590820, -2308.9277344, 1978.9603271, -3951.7392578, 4020.4868164
1: -1583.4219971, 1677.7437744, -1859.4763184, 1938.2121582, -3521.6342773, 3537.2194824
2: -2322.5876465, 1817.8847656, -2747.4638672, 2102.9970703, -4425.5844727, 4565.3476562
3: -897.7653809, 2353.9465332, -1039.7521973, 2749.6503906, -3647.4155273, 3393.6984863
4: -2557.8483887, 1770.4099121, -3023.1081543, 2047.6184082, -4605.4658203, 4793.5180664

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.7862918, upper bound: 3613.9233561
time: 0.68 seconds

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.7862918, upper bound: 3613.9245938
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -2120.2116699, 1846.7767334, -2463.8974609, 2115.7761230, -4235.9877930, 4310.6743164
1: -1701.8973389, 1811.7041016, -1984.0234375, 2072.2014160, -3774.0986328, 3795.7275391
2: -2495.2231445, 1961.5872803, -2928.6911621, 2246.4538574, -4741.6767578, 4890.2773438
3: -966.3668823, 2534.0012207, -1107.6962891, 2935.7224121, -3902.0888672, 3641.6972656
4: -2747.3745117, 1910.0290527, -3223.6826172, 2189.1254883, -4936.4995117, 5133.7104492

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8425709, upper bound: 3613.9233500
time: 0.64 seconds

## Relational analysis of NS_A1_B2_B2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8425709, upper bound: 3613.9245073
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2108.3735352, 1833.3608398, -2374.0385742, 2035.3465576, -4143.7202148, 4207.3994141
1: -1692.4837646, 1798.4423828, -1912.6656494, 1994.0113525, -3686.4951172, 3711.1079102
2: -2481.3803711, 1947.3975830, -2825.5678711, 2163.0231934, -4644.4033203, 4772.9653320
3: -960.4921875, 2518.5825195, -1069.0500488, 2828.6394043, -3789.1315918, 3587.6325684
4: -2732.0664062, 1895.8527832, -3108.4445801, 2106.2158203, -4838.2807617, 5004.2973633

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8425709, upper bound: 3613.9233561
time: 0.73 seconds

## Relational analysis of NS_A1_B2_B2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8425709, upper bound: 3613.9245012
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2075.3298340, 1778.9835205, -1922.9985352, 1672.3048096, -3747.6342773, 3701.9819336
1: -1671.2916260, 1740.2213135, -1543.5406494, 1639.6268311, -3310.9184570, 3283.7619629
2: -2469.9309082, 1889.6323242, -2264.1687012, 1776.6118164, -4246.5424805, 4153.8007812
3: -932.2543335, 2464.4843750, -876.2282104, 2296.4216309, -3228.6760254, 3340.7124023
4: -2716.8002930, 1840.9689941, -2493.2343750, 1730.2576904, -4447.0581055, 4334.2031250

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.3263419, upper bound: 3611.9134615
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.3263419, upper bound: 3612.0913295
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -2326.8193359, 1997.6417236, -1886.7838135, 1640.3638916, -3967.1823730, 3884.4255371
1: -1874.2145996, 1956.2360840, -1513.6895752, 1609.1557617, -3483.3698730, 3469.9252930
2: -2767.0754395, 2122.2719727, -2219.9250488, 1743.9522705, -4511.0278320, 4342.1962891
3: -1046.6068115, 2771.1145020, -859.4227295, 2250.0146484, -3296.6213379, 3630.5368652
4: -3044.5729980, 2067.9660645, -2443.6701660, 1695.8272705, -4740.3994141, 4511.6357422

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5561045, upper bound: 3612.5936805
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0217443, upper bound: 3612.5952942
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -2326.7902832, 1997.6098633, -2088.8212891, 1820.4105225, -4147.2006836, 4086.4309082
1: -1874.1875000, 1956.2061768, -1676.7429199, 1785.9445801, -3660.1318359, 3632.9492188
2: -2767.0207520, 2122.2380371, -2458.2387695, 1933.5567627, -4700.5766602, 4580.4765625
3: -1046.5935059, 2771.0688477, -952.1087646, 2497.4025879, -3543.9956055, 3723.1777344
4: -3044.5139160, 2067.9301758, -2706.6801758, 1882.7955322, -4927.3095703, 4774.6098633

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A1

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5589180, upper bound: 3612.5936805
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A2

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0223578, upper bound: 3612.5952942
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -2478.1323242, 2144.8574219, -2082.4411621, 1816.4946289, -4294.6269531, 4227.2973633
1: -1996.0679932, 2101.0026855, -1672.2194824, 1782.3671875, -3778.4350586, 3773.2221680
2: -2949.6411133, 2277.8205566, -2452.7009277, 1929.6461182, -4879.2871094, 4730.5214844
3: -1118.4505615, 2960.7971191, -950.3276367, 2491.3442383, -3609.7949219, 3911.1247559
4: -3244.9345703, 2222.6145020, -2699.4194336, 1879.0957031, -5124.0302734, 4922.0341797

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_A1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0043742, upper bound: 3611.5828355
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A2_A1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_A1_A1_B1_B1

### Relational analysis result of NS_A2_B1_A2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4734865, upper bound: 3611.6775390
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A2_A1_A1_B1_B2

### Relational analysis result of NS_A2_B1_A2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4734865, upper bound: 3611.7702119
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -2478.0554199, 2144.8020020, -2111.0319824, 1836.8913574, -4314.9462891, 4255.8339844
1: -1996.0053711, 2100.9492188, -1694.9891357, 1802.7271729, -3798.7324219, 3795.9379883
2: -2949.5437012, 2277.7631836, -2485.4467773, 1952.0012207, -4901.5449219, 4763.2094727
3: -1118.4224854, 2960.7084961, -962.4727173, 2524.1877441, -3642.6103516, 3923.1809082
4: -3244.8286133, 2222.5578613, -2735.7641602, 1899.9638672, -5144.7910156, 4958.3212891

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_A1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0043742, upper bound: 3612.7422704
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A2_A1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_A1_A1_B2_B1

### Relational analysis result of NS_A2_B1_A2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4734865, upper bound: 3611.9221605
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A2_A1_A1_B2_B2

### Relational analysis result of NS_A2_B1_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4734865, upper bound: 3612.7424471
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -2494.9543457, 2154.3381348, -2082.4411621, 1816.4946289, -4311.4482422, 4236.7773438
1: -2010.4550781, 2109.9956055, -1672.2194824, 1782.3671875, -3792.8222656, 3782.2150879
2: -2970.7456055, 2287.4833984, -2452.7009277, 1929.6461182, -4900.3916016, 4740.1845703
3: -1125.5938721, 2981.4509277, -950.3276367, 2491.3442383, -3616.9377441, 3931.7785645
4: -3267.1584473, 2231.9443359, -2699.4194336, 1879.0957031, -5146.2539062, 4931.3637695

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_A1_A2_B1_B1

### Relational analysis result of NS_A2_B1_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5144836, upper bound: 3611.6777285
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A2_A1_A2_B1_B2

### Relational analysis result of NS_A2_B1_A2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5144836, upper bound: 3611.7704042
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -2494.8796387, 2154.2841797, -2111.0319824, 1836.8913574, -4331.7709961, 4265.3154297
1: -2010.3940430, 2109.9438477, -1694.9891357, 1802.7271729, -3813.1210938, 3804.9328613
2: -2970.6511230, 2287.4274902, -2485.4467773, 1952.0012207, -4922.6518555, 4772.8735352
3: -1125.5662842, 2981.3649902, -962.4727173, 2524.1877441, -3649.7539062, 3943.8376465
4: -3267.0544434, 2231.8879395, -2735.7641602, 1899.9638672, -5167.0180664, 4967.6523438

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B1

### Relational analysis result of NS_A2_B1_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5144836, upper bound: 3611.9223468
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B2

### Relational analysis result of NS_A2_B1_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5144836, upper bound: 3612.6387800
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2397.9648438, 2057.9523926, -1987.2487793, 1727.2838135, -4125.2485352, 4045.2009277
1: -1930.3010254, 2014.5875244, -1594.8696289, 1693.3439941, -3623.6447754, 3609.4570312
2: -2849.9020996, 2184.5903320, -2339.1728516, 1834.6950684, -4684.5961914, 4523.7631836
3: -1077.5959473, 2855.4157715, -905.3850098, 2372.7167969, -3450.3127441, 3760.8005371
4: -3137.6254883, 2129.1381836, -2576.0402832, 1786.7614746, -4924.3867188, 4705.1787109

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.9233500, upper bound: 3611.7862918
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.9233500, upper bound: 3611.8928309
time: 1.83 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2308.9277344, 1978.9603271, -1972.7789307, 1711.5590820, -4020.4868164, 3951.7392578
1: -1859.4763184, 1938.2121582, -1583.4219971, 1677.7437744, -3537.2192383, 3521.6340332
2: -2747.4638672, 2102.9970703, -2322.5876465, 1817.8847656, -4565.3476562, 4425.5844727
3: -1039.7521973, 2749.6503906, -897.7653809, 2353.9465332, -3393.6987305, 3647.4155273
4: -3023.1081543, 2047.6184082, -2557.8483887, 1770.4099121, -4793.5180664, 4605.4658203

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.9233500, upper bound: 3611.7862918
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.9233500, upper bound: 3611.9409444
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2463.8974609, 2115.7763672, -2120.2116699, 1846.7767334, -4310.6743164, 4235.9877930
1: -1984.0234375, 2072.2016602, -1701.8973389, 1811.7041016, -3795.7275391, 3774.0983887
2: -2928.6911621, 2246.4541016, -2495.2231445, 1961.5872803, -4890.2773438, 4741.6772461
3: -1107.6962891, 2935.7224121, -966.3668823, 2534.0012207, -3641.6975098, 3902.0888672
4: -3223.6826172, 2189.1254883, -2747.3745117, 1910.0290527, -5133.7109375, 4936.5000000

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.9233500, upper bound: 3611.8425709
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.9233500, upper bound: 3611.8425709
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2374.0385742, 2035.3465576, -2108.3735352, 1833.3608398, -4207.3994141, 4143.7202148
1: -1912.6656494, 1994.0113525, -1692.4837646, 1798.4423828, -3711.1079102, 3686.4951172
2: -2825.5678711, 2163.0231934, -2481.3803711, 1947.3975830, -4772.9653320, 4644.4033203
3: -1069.0500488, 2828.6394043, -960.4921875, 2518.5825195, -3587.6325684, 3789.1315918
4: -3108.4445801, 2106.2158203, -2732.0664062, 1895.8527832, -5004.2973633, 4838.2802734

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.9233500, upper bound: 3611.8425709
time: 1.51 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.9233500, upper bound: 3611.8425709
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -2016.1315918, 1753.1184082, -4040.9772949, 3643.7629395, -5494.3383789, 5794.0952148
1: -1624.1541748, 1721.8516846, -3258.2163086, 3586.9877930, -5073.3242188, 4980.0678711
2: -2415.1628418, 1862.4095459, -4915.1596680, 3856.4182129, -6123.1191406, 6761.8125000
3: -912.2088013, 2434.4267578, -1809.5748291, 4956.0493164, -5819.6381836, 4208.6577148
4: -2654.6584473, 1815.8958740, -5376.3759766, 3746.6291504, -6244.6494141, 7188.2954102

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_A1_A2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2184311, upper bound: 3613.2901708
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_A2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3780831, upper bound: 3613.2903104
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3858.8288574, 3476.9750977, -6013.2416992, 5519.2744141, -8739.5244141, 9112.6201172
1: -3110.0395508, 3427.8459473, -4842.5952148, 5451.9565430, -8018.0224609, 7947.5551758
2: -4695.0322266, 3685.5327148, -7358.2250977, 5860.1455078, -9851.8779297, 10516.2832031
3: -1728.6866455, 4724.4501953, -2703.7177734, 7426.7338867, -8825.3378906, 7125.1030273
4: -5128.0468750, 3577.4528809, -8027.0366211, 5675.7690430, -10098.7080078, 11073.7714844

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9196269, upper bound: 3612.9208430
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9199986, upper bound: 3613.0074971
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -6025.1059570, 5531.3374023, -6025.1059570, 5531.3374023, -10777.6152344, 10777.6152344
1: -4852.1328125, 5463.9526367, -4852.1328125, 5463.9526367, -9651.5029297, 9651.5029297
2: -7372.8691406, 5872.9902344, -7372.8691406, 5872.9902344, -12303.2919922, 12303.2919922
3: -2709.3872070, 7442.1484375, -2709.3872070, 7442.1484375, -9661.6455078, 9661.6455078
4: -8043.0620117, 5688.1357422, -8043.0620117, 5688.1357422, -12783.0449219, 12783.0458984

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9211812, upper bound: 3613.2221689
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9354275, upper bound: 3613.2894804
time: 0.87 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 2.93 seconds
NS_A1_B2_B2_B1_B1_A1_B1, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -3611.7581409, upper bound: 3611.3824795
NS_A1_B2_B2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3611.7599242, upper bound: 3613.4736060
NS_A1_B2_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3611.6775390, upper bound: 3613.4734918
NS_A1_B2_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3611.6775390, upper bound: 3613.4734865
NS_A1_B2_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3611.6775390, upper bound: 3613.5144836
NS_A1_B2_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3611.6775390, upper bound: 3613.5145514
NS_A1_B2_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3611.6775390, upper bound: 3613.5143700
NS_A1_B2_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3611.6775390, upper bound: 3613.5146042
NS_A1_B2_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3611.7862918, upper bound: 3613.9233561
NS_A1_B2_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3611.7862918, upper bound: 3613.9245999
NS_A1_B2_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3611.7862918, upper bound: 3613.9233561
NS_A1_B2_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3611.7862918, upper bound: 3613.9245938
NS_A1_B2_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3611.8425709, upper bound: 3613.9233500
NS_A1_B2_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3611.8425709, upper bound: 3613.9245073
NS_A1_B2_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3611.8425709, upper bound: 3613.9233561
NS_A1_B2_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3611.8425709, upper bound: 3613.9245012
NS_A2_B1_A1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -3612.3263419, upper bound: 3611.9134615
NS_A2_B1_A1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -3612.3263419, upper bound: 3612.0913295
NS_A2_B1_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3613.5561045, upper bound: 3612.5936805
NS_A2_B1_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3614.0217443, upper bound: 3612.5952942
NS_A2_B1_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3613.5589180, upper bound: 3612.5936805
NS_A2_B1_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3614.0223578, upper bound: 3612.5952942
NS_A2_B1_A2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3613.4734865, upper bound: 3611.6775390
NS_A2_B1_A2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3613.4734865, upper bound: 3611.7702119
NS_A2_B1_A2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3613.4734865, upper bound: 3611.9221605
NS_A2_B1_A2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3613.4734865, upper bound: 3612.7424471
NS_A2_B1_A2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3613.5144836, upper bound: 3611.6777285
NS_A2_B1_A2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3613.5144836, upper bound: 3611.7704042
NS_A2_B1_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3613.5144836, upper bound: 3611.9223468
NS_A2_B1_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3613.5144836, upper bound: 3612.6387800
NS_A2_B1_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3613.9233500, upper bound: 3611.7862918
NS_A2_B1_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3613.9233500, upper bound: 3611.8928309
NS_A2_B1_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3613.9233500, upper bound: 3611.7862918
NS_A2_B1_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3613.9233500, upper bound: 3611.9409444
NS_A2_B1_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3613.9233500, upper bound: 3611.8425709
NS_A2_B1_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3613.9233500, upper bound: 3611.8425709
NS_A2_B1_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3613.9233500, upper bound: 3611.8425709
NS_A2_B1_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3613.9233500, upper bound: 3611.8425709
NS_A2_B2_A1_B2_A1_A2_A1, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -3613.2184311, upper bound: 3613.2901708
NS_A2_B2_A1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -3613.3780831, upper bound: 3613.2903104
NS_A2_B2_A1_B2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -3612.9196269, upper bound: 3612.9208430
NS_A2_B2_A1_B2_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -3612.9199986, upper bound: 3613.0074971
NS_A2_B2_A2_B2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -3612.9211812, upper bound: 3613.2221689
NS_A2_B2_A2_B2_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 2.93
Output dim: 0, lower bound: -3612.9354275, upper bound: 3613.2894804

## BFS NS instance: NS_A1_B2_B2_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -2082.4411621, 1816.4946289, -2474.2082520, 2141.5664062, -4224.0063477, 4290.7031250
1: -1672.2194824, 1782.3671875, -1992.8997803, 2097.8151855, -3770.0344238, 3775.2668457
2: -2452.7009277, 1929.6461182, -2945.0317383, 2274.3681641, -4727.0693359, 4874.6777344
3: -950.3276367, 2491.3442383, -1116.7445068, 2956.1762695, -3906.5039062, 3608.0886230
4: -2699.4194336, 1879.0957031, -3239.8596191, 2219.2412109, -4918.6606445, 5118.9550781

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_B2_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B2_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_B2_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_B1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_B2_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.7403670, upper bound: 3613.4711716
time: 1.06 seconds

## Relational analysis of NS_A1_B2_B2_B1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_B2_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.7403670, upper bound: 3613.4735986
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_B2_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -2198.3762207, 1923.3669434, -2478.0427246, 2144.7927246, -4343.1689453, 4401.4096680
1: -1764.6110840, 1887.4392090, -1995.9945068, 2100.9401855, -3865.5512695, 3883.4335938
2: -2587.5109863, 2042.6828613, -2949.5249023, 2277.7539062, -4865.2646484, 4992.2065430
3: -1003.2032471, 2632.3742676, -1118.4180908, 2960.6928711, -3963.8959961, 3750.7919922
4: -2848.5363770, 1990.8115234, -3244.8078613, 2222.5471191, -5071.0830078, 5235.6191406

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_B2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_B2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_B1_B1_A2_A1_B1

### Relational analysis result of NS_A1_B2_B2_B1_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.9132242, upper bound: 3611.3824673
time: 0.76 seconds

## Relational analysis of NS_A1_B2_B2_B1_B1_A2_A1_B2

### Relational analysis result of NS_A1_B2_B2_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.9150142, upper bound: 3613.4734865
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_B2_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -2119.1289062, 1843.8908691, -2478.0637207, 2144.8081055, -4263.9370117, 4321.9545898
1: -1701.5185547, 1809.7534180, -1996.0123291, 2100.9553223, -3802.4736328, 3805.7656250
2: -2495.1850586, 1959.6323242, -2949.5539551, 2277.7695312, -4772.9545898, 4909.1865234
3: -966.2225342, 2534.0144043, -1118.4256592, 2960.7182617, -3926.9409180, 3652.4399414
4: -2746.3649902, 1907.2337646, -3244.8398438, 2222.5639648, -4968.9287109, 5152.0737305

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_B2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_B2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_B1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B2_B2_B1_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.9132242, upper bound: 3611.3831078
time: 0.73 seconds

## Relational analysis of NS_A1_B2_B2_B1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B2_B2_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.9150142, upper bound: 3613.4737078
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_B2_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -2174.8547363, 1907.8500977, -2494.9409180, 2154.3281250, -4329.1826172, 4402.7910156
1: -1745.8293457, 1872.0988770, -2010.4438477, 2109.9860840, -3855.8154297, 3882.5427246
2: -2560.6362305, 2026.8896484, -2970.7260742, 2287.4731445, -4848.1093750, 4997.6157227
3: -993.0927124, 2605.6098633, -1125.5893555, 2981.4353027, -3974.5280762, 3731.1984863
4: -2818.8544922, 1974.6855469, -3267.1372070, 2231.9340820, -5050.7885742, 5241.8227539

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B2_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B2_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_B2_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_B2_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_B1_B2_A1_A1_B1

### Relational analysis result of NS_A1_B2_B2_B1_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.6749365, upper bound: 3611.3933792
time: 0.64 seconds

## Relational analysis of NS_A1_B2_B2_B1_B2_A1_A1_B2

### Relational analysis result of NS_A1_B2_B2_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.6769015, upper bound: 3613.5145017
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_B2_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -2089.4301758, 1822.5927734, -2494.9614258, 2154.3432617, -4243.7729492, 4317.5541992
1: -1677.8387451, 1788.5040283, -2010.4609375, 2110.0004883, -3787.8393555, 3798.9648438
2: -2461.1071777, 1936.3327637, -2970.7543945, 2287.4885254, -4748.5957031, 4907.0869141
3: -953.5634155, 2499.8732910, -1125.5964355, 2981.4599609, -3935.0234375, 3625.4697266
4: -2708.5678711, 1885.4360352, -3267.1677246, 2231.9497070, -4940.5175781, 5152.6030273

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B2_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_B2_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_B2_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_B1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B2_B2_B1_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.6749365, upper bound: 3611.3933880
time: 0.74 seconds

## Relational analysis of NS_A1_B2_B2_B1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B2_B2_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.6769015, upper bound: 3613.5145458
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_B2_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -2198.3762207, 1923.3669434, -2494.8696289, 2154.2770996, -4352.6533203, 4418.2363281
1: -1764.6110840, 1887.4392090, -2010.3859863, 2109.9367676, -3874.5478516, 3897.8251953
2: -2587.5109863, 2042.6828613, -2970.6359863, 2287.4199219, -4874.9306641, 5013.3178711
3: -1003.2032471, 2632.3742676, -1125.5632324, 2981.3535156, -3984.5566406, 3757.9375000
4: -2848.5363770, 1990.8115234, -3267.0388184, 2231.8808594, -5080.4169922, 5257.8500977

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B2_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B2_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_B2_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_B2_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_B1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_B2_B1_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.9132261, upper bound: 3611.3933419
time: 0.67 seconds

## Relational analysis of NS_A1_B2_B2_B1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_B2_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.9152004, upper bound: 3613.5143632
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_B2_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -2119.1289062, 1843.8908691, -2494.8911133, 2154.2927246, -4273.4218750, 4338.7812500
1: -1701.5185547, 1809.7534180, -2010.4034424, 2109.9519043, -3811.4704590, 3820.1567383
2: -2495.1850586, 1959.6323242, -2970.6655273, 2287.4360352, -4782.6210938, 4930.2978516
3: -966.2225342, 2534.0144043, -1125.5705566, 2981.3786621, -3947.6010742, 3659.5849609
4: -2746.3649902, 1907.2337646, -3267.0703125, 2231.8972168, -4978.2622070, 5174.3041992

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_B2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_B2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_B1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_B2_B1_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.9132261, upper bound: 3611.3933442
time: 0.77 seconds

## Relational analysis of NS_A1_B2_B2_B1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_B2_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.9152004, upper bound: 3613.5143632
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1977.5880127, 1717.8486328, -2397.8933105, 2057.8969727, -4035.4848633, 4115.7416992
1: -1586.3464355, 1683.7786865, -1930.2419434, 2014.5339355, -3600.8803711, 3614.0202637
2: -2325.9230957, 1823.9427490, -2849.8095703, 2184.5322266, -4510.4550781, 4673.7514648
3: -899.5002441, 2358.9836426, -1077.5671387, 2855.3312988, -3754.8315430, 3436.5507812
4: -2562.2343750, 1776.0528564, -3137.5258789, 2129.0795898, -4691.3139648, 4913.5771484

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_B2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.7862835, upper bound: 3614.0164730
time: 0.72 seconds

## Relational analysis of NS_A1_B2_B2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_B2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.7862582, upper bound: 3614.0170017
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1943.8889160, 1682.8842773, -2397.8620605, 2057.8745117, -4001.7634277, 4080.7463379
1: -1560.3826904, 1649.3551025, -1930.2169189, 2014.5123291, -3574.8950195, 3579.5720215
2: -2288.9770508, 1787.2509766, -2849.7702637, 2184.5087891, -4473.4843750, 4637.0205078
3: -883.4014282, 2317.5876465, -1077.5557861, 2855.2958984, -3738.6972656, 3395.1430664
4: -2520.9567871, 1740.5045166, -3137.4826660, 2129.0568848, -4650.0131836, 4877.9863281

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_B2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.7862835, upper bound: 3614.0164681
time: 0.63 seconds

## Relational analysis of NS_A1_B2_B2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_B2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.7862582, upper bound: 3614.0169967
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1977.5880127, 1717.8486328, -2308.9147949, 1978.9503174, -3956.5383301, 4026.7634277
1: -1586.3464355, 1683.7786865, -1859.4660645, 1938.2023926, -3524.5488281, 3543.2446289
2: -2325.9230957, 1823.9427490, -2747.4492188, 2102.9870605, -4428.9101562, 4571.3920898
3: -899.5002441, 2358.9836426, -1039.7465820, 2749.6359863, -3649.1362305, 3398.7302246
4: -2562.2343750, 1776.0528564, -3023.0925293, 2047.6083984, -4609.8422852, 4799.1440430

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_B2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.7862806, upper bound: 3613.9232770
time: 0.68 seconds

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_B2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.7862582, upper bound: 3613.9232770
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1943.8889160, 1682.8842773, -2308.8920898, 1978.9335938, -3922.8222656, 3991.7763672
1: -1560.3826904, 1649.3551025, -1859.4475098, 1938.1861572, -3498.5688477, 3508.8022461
2: -2288.9770508, 1787.2509766, -2747.4194336, 2102.9692383, -4391.9453125, 4534.6704102
3: -883.4014282, 2317.5876465, -1039.7382812, 2749.6091309, -3633.0104980, 3357.3256836
4: -2520.9567871, 1740.5045166, -3023.0595703, 2047.5906982, -4568.5468750, 4763.5639648

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_B2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.7862806, upper bound: 3613.9244825
time: 0.68 seconds

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_B2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.7862582, upper bound: 3613.9244825
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2159.6970215, 1876.9660645, -2463.8439941, 2115.7326660, -4275.4296875, 4340.8095703
1: -1732.6105957, 1840.3956299, -1983.9792480, 2072.1596680, -3804.7692871, 3824.3750000
2: -2538.4934082, 1992.0585938, -2928.6206055, 2246.4086914, -4784.9008789, 4920.6787109
3: -981.9328613, 2576.9663086, -1107.6740723, 2935.6577148, -3917.5900879, 3684.6401367
4: -2796.5915527, 1940.0974121, -3223.6059570, 2189.0800781, -4985.6718750, 5163.7031250

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_B2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8425626, upper bound: 3614.0164505
time: 0.76 seconds

## Relational analysis of NS_A1_B2_B2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_B2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8425372, upper bound: 3614.0169971
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2079.5249023, 1804.6406250, -2463.7961426, 2115.6982422, -4195.2231445, 4268.4360352
1: -1669.4020996, 1770.1657715, -1983.9399414, 2072.1259766, -3741.5278320, 3754.1057129
2: -2447.5773926, 1916.9589844, -2928.5605469, 2246.3728027, -4693.9501953, 4845.5195312
3: -947.0042114, 2482.3625488, -1107.6562500, 2935.6022949, -3882.6062012, 3590.0185547
4: -2694.8488770, 1865.7957764, -3223.5397949, 2189.0444336, -4883.8935547, 5089.3344727

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_B2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8425626, upper bound: 3614.0164505
time: 0.78 seconds

## Relational analysis of NS_A1_B2_B2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_B2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8425372, upper bound: 3614.0169921
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2159.6970215, 1876.9660645, -2374.0427246, 2035.3471680, -4195.0439453, 4251.0083008
1: -1732.6105957, 1840.3956299, -1912.6689453, 1994.0125732, -3726.6225586, 3753.0644531
2: -2538.4934082, 1992.0585938, -2825.5732422, 2163.0234375, -4701.5161133, 4817.6318359
3: -981.9328613, 2576.9663086, -1069.0504150, 2828.6435547, -3810.5756836, 3646.0166016
4: -2796.5915527, 1940.0974121, -3108.4506836, 2106.2163086, -4902.8076172, 5048.5478516

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_B2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8425598, upper bound: 3613.9232603
time: 0.86 seconds

## Relational analysis of NS_A1_B2_B2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_B2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8425372, upper bound: 3613.9232603
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2079.5249023, 1804.6406250, -2374.0029297, 2035.3190918, -4114.8437500, 4178.6435547
1: -1669.4020996, 1770.1657715, -1912.6363525, 1993.9849854, -3663.3869629, 3682.8022461
2: -2447.5773926, 1916.9589844, -2825.5229492, 2162.9951172, -4610.5722656, 4742.4819336
3: -947.0042114, 2482.3625488, -1069.0357666, 2828.5976562, -3775.6018066, 3551.3984375
4: -2694.8488770, 1865.7957764, -3108.3955078, 2106.1875000, -4801.0361328, 4974.1909180

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_B2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8425598, upper bound: 3613.9238978
time: 0.73 seconds

## Relational analysis of NS_A1_B2_B2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_B2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8425372, upper bound: 3613.9238978
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -2246.5732422, 1929.6694336, -1886.7838135, 1640.3638916, -3886.9360352, 3816.4531250
1: -1809.9316406, 1890.0814209, -1513.6895752, 1609.1557617, -3419.0871582, 3403.7709961
2: -2674.1757812, 2050.4313965, -2219.9250488, 1743.9522705, -4418.1279297, 4270.3564453
3: -1010.8899536, 2677.4301758, -859.4227295, 2250.0146484, -3260.9045410, 3536.8525391
4: -2941.9868164, 1997.9365234, -2443.6701660, 1695.8272705, -4637.8125000, 4441.6064453

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5554447, upper bound: 3612.5875344
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5554447, upper bound: 3612.5936792
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -2268.9897461, 1947.3017578, -1874.7276611, 1630.3983154, -3899.3879395, 3822.0292969
1: -1828.8757324, 1906.6157227, -1504.1112061, 1599.4117432, -3428.2875977, 3410.7270508
2: -2702.1101074, 2068.5043945, -2206.0766602, 1733.4128418, -4435.5224609, 4274.5810547
3: -1020.8913574, 2705.8554688, -854.1786499, 2236.1303711, -3257.0212402, 3560.0341797
4: -2971.7778320, 2015.8797607, -2428.2998047, 1685.5743408, -4657.3520508, 4444.1796875

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A2_A1

### Relational analysis result of NS_A2_B1_A1_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7943513, upper bound: 3612.5661049
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A2_A2

### Relational analysis result of NS_A2_B1_A1_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0217443, upper bound: 3612.5662630
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -2246.5375977, 1929.6330566, -2088.8212891, 1820.4105225, -4066.9475098, 4018.4543457
1: -1809.8990479, 1890.0465088, -1676.7429199, 1785.9445801, -3595.8437500, 3566.7893066
2: -2674.1130371, 2050.3923340, -2458.2387695, 1933.5567627, -4607.6689453, 4508.6308594
3: -1010.8740234, 2677.3771973, -952.1087646, 2497.4025879, -3508.2763672, 3629.4858398
4: -2941.9179688, 1997.8968506, -2706.6801758, 1882.7955322, -4824.7133789, 4704.5766602

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A1_A1

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5319262, upper bound: 3612.5790557
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A1_A2

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5319262, upper bound: 3612.5414763
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -2268.9514160, 1947.2677002, -2077.6520996, 1811.1444092, -4080.0957031, 4024.9191895
1: -1828.8417969, 1906.5833740, -1667.8284912, 1776.8822021, -3605.7236328, 3574.4118652
2: -2702.0461426, 2068.4685059, -2445.3176270, 1923.7659912, -4625.8120117, 4513.7856445
3: -1020.8768311, 2705.8049316, -947.1780396, 2484.4365234, -3505.3134766, 3652.9829102
4: -2971.7092285, 2015.8424072, -2692.3569336, 1873.2725830, -4844.9794922, 4708.1977539

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A2_A1

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7944839, upper bound: 3612.5661049
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A2_A2

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0223578, upper bound: 3612.5662630
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -2478.1184082, 2144.8474121, -2174.8547363, 1907.8500977, -4385.9682617, 4319.7016602
1: -1996.0566406, 2100.9929199, -1745.8293457, 1872.0988770, -3868.1552734, 3846.8220215
2: -2949.6210938, 2277.8098145, -2560.6362305, 2026.8896484, -4976.5107422, 4838.4462891
3: -1118.4458008, 2960.7810059, -993.0927124, 2605.6098633, -3724.0556641, 3953.8737793
4: -3244.9143066, 2222.6037598, -2818.8544922, 1974.6855469, -5219.5996094, 5041.4580078

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_A1_A1_B1_B1_A1

### Relational analysis result of NS_A2_B1_A2_A1_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.3824673, upper bound: 3611.6749334
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A2_A1_A1_B1_B1_A2

### Relational analysis result of NS_A2_B1_A2_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4735125, upper bound: 3611.6767121
time: 0.61 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -2478.1398926, 2144.8625488, -2089.4301758, 1822.5927734, -4300.7324219, 4234.2929688
1: -1996.0738525, 2101.0075684, -1677.8387451, 1788.5040283, -3784.5776367, 3778.8461914
2: -2949.6501465, 2277.8261719, -2461.1071777, 1936.3327637, -4885.9829102, 4738.9331055
3: -1118.4533691, 2960.8054199, -953.5634155, 2499.8732910, -3618.3266602, 3914.3688965
4: -3244.9453125, 2222.6201172, -2708.5678711, 1885.4360352, -5130.3808594, 4931.1879883

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_A1_A1_B1_B2_A1

### Relational analysis result of NS_A2_B1_A2_A1_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.3824673, upper bound: 3611.7581381
time: 0.59 seconds

## Relational analysis of NS_A2_B1_A2_A1_A1_B1_B2_A2

### Relational analysis result of NS_A2_B1_A2_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4735125, upper bound: 3611.7599242
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -2478.0427246, 2144.7927246, -2198.3762207, 1923.3669434, -4401.4096680, 4343.1689453
1: -1995.9945068, 2100.9401855, -1764.6110840, 1887.4392090, -3883.4335938, 3865.5512695
2: -2949.5249023, 2277.7539062, -2587.5109863, 2042.6828613, -4992.2065430, 4865.2641602
3: -1118.4180908, 2960.6928711, -1003.2032471, 2632.3742676, -3750.7922363, 3963.8959961
4: -3244.8078613, 2222.5471191, -2848.5363770, 1990.8115234, -5235.6191406, 5071.0830078

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_A1_A1_B2_B1_A1

### Relational analysis result of NS_A2_B1_A2_A1_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.3824673, upper bound: 3611.9132242
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A2_A1_A1_B2_B1_A2

### Relational analysis result of NS_A2_B1_A2_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4734865, upper bound: 3611.9150142
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -2478.0637207, 2144.8081055, -2119.1289062, 1843.8908691, -4321.9541016, 4263.9370117
1: -1996.0122070, 2100.9553223, -1701.5185547, 1809.7534180, -3805.7656250, 3802.4736328
2: -2949.5537109, 2277.7695312, -2495.1850586, 1959.6323242, -4909.1860352, 4772.9545898
3: -1118.4256592, 2960.7182617, -966.2225342, 2534.0144043, -3652.4399414, 3926.9406738
4: -3244.8398438, 2222.5639648, -2746.3649902, 1907.2337646, -5152.0737305, 4968.9282227

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_A1_A1_B2_B2_A1

### Relational analysis result of NS_A2_B1_A2_A1_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.3824673, upper bound: 3612.7374182
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A2_A1_A1_B2_B2_A2

### Relational analysis result of NS_A2_B1_A2_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4734865, upper bound: 3612.7354297
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -2494.9409180, 2154.3281250, -2174.8547363, 1907.8500977, -4402.7910156, 4329.1826172
1: -2010.4438477, 2109.9860840, -1745.8293457, 1872.0988770, -3882.5427246, 3855.8154297
2: -2970.7260742, 2287.4731445, -2560.6362305, 2026.8896484, -4997.6157227, 4848.1093750
3: -1125.5893555, 2981.4353027, -993.0927124, 2605.6098633, -3731.1987305, 3974.5280762
4: -3267.1372070, 2231.9340820, -2818.8544922, 1974.6855469, -5241.8227539, 5050.7885742

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_A1_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_A2_A1_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.3933792, upper bound: 3611.6749365
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A2_A1_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_A2_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5144961, upper bound: 3611.6769015
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -2494.9614258, 2154.3432617, -2089.4301758, 1822.5927734, -4317.5541992, 4243.7729492
1: -2010.4606934, 2110.0004883, -1677.8387451, 1788.5040283, -3798.9648438, 3787.8393555
2: -2970.7543945, 2287.4885254, -2461.1071777, 1936.3327637, -4907.0869141, 4748.5957031
3: -1125.5964355, 2981.4599609, -953.5634155, 2499.8732910, -3625.4697266, 3935.0234375
4: -3267.1677246, 2231.9497070, -2708.5678711, 1885.4360352, -5152.6030273, 4940.5175781

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_A1_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_A2_A1_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.3933792, upper bound: 3611.7581421
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A2_A1_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_A2_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5144961, upper bound: 3611.7601164
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -2494.8669434, 2154.2746582, -2198.3762207, 1923.3669434, -4418.2338867, 4352.6508789
1: -2010.3834229, 2109.9348145, -1764.6110840, 1887.4392090, -3897.8225098, 3874.5458984
2: -2970.6318359, 2287.4172363, -2587.5109863, 2042.6828613, -5013.3139648, 4874.9282227
3: -1125.5618896, 2981.3496094, -1003.2032471, 2632.3742676, -3757.9360352, 3984.5527344
4: -3267.0341797, 2231.8781738, -2848.5363770, 1990.8115234, -5257.8457031, 5080.4135742

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B1_A1

### Relational analysis result of NS_A2_B1_A2_A1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.3933793, upper bound: 3611.9131581
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B1_A2

### Relational analysis result of NS_A2_B1_A2_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5144836, upper bound: 3611.9152004
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -2494.8879395, 2154.2902832, -2119.1289062, 1843.8908691, -4338.7778320, 4273.4189453
1: -2010.4008789, 2109.9494629, -1701.5185547, 1809.7534180, -3820.1542969, 3811.4680176
2: -2970.6608887, 2287.4333496, -2495.1850586, 1959.6323242, -4930.2929688, 4782.6181641
3: -1125.5693359, 2981.3747559, -966.2225342, 2534.0144043, -3659.5837402, 3947.5971680
4: -3267.0656738, 2231.8947754, -2746.3649902, 1907.2337646, -5174.2993164, 4978.2597656

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B2_A1

### Relational analysis result of NS_A2_B1_A2_A1_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.3933793, upper bound: 3612.5749284
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B2_A2

### Relational analysis result of NS_A2_B1_A2_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5144836, upper bound: 3612.6272545
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -2397.8933105, 2057.8969727, -1977.5880127, 1717.8486328, -4115.7421875, 4035.4848633
1: -1930.2419434, 2014.5339355, -1586.3464355, 1683.7786865, -3614.0202637, 3600.8803711
2: -2849.8095703, 2184.5322266, -2325.9230957, 1823.9427490, -4673.7514648, 4510.4550781
3: -1077.5671387, 2855.3312988, -899.5002441, 2358.9836426, -3436.5507812, 3754.8315430
4: -3137.5258789, 2129.0795898, -2562.2343750, 1776.0528564, -4913.5771484, 4691.3139648

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0164681, upper bound: 3611.7862835
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0169967, upper bound: 3611.7862582
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -2397.8620605, 2057.8745117, -1943.8889160, 1682.8842773, -4080.7463379, 4001.7634277
1: -1930.2167969, 2014.5123291, -1560.3826904, 1649.3551025, -3579.5717773, 3574.8950195
2: -2849.7702637, 2184.5090332, -2288.9770508, 1787.2509766, -4637.0209961, 4473.4843750
3: -1077.5557861, 2855.2956543, -883.4014282, 2317.5876465, -3395.1430664, 3738.6970215
4: -3137.4826660, 2129.0566406, -2520.9567871, 1740.5045166, -4877.9863281, 4650.0131836

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0164681, upper bound: 3611.8926687
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0169967, upper bound: 3611.8927471
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -2308.9147949, 1978.9503174, -1977.5880127, 1717.8486328, -4026.7634277, 3956.5383301
1: -1859.4660645, 1938.2023926, -1586.3464355, 1683.7786865, -3543.2446289, 3524.5488281
2: -2747.4492188, 2102.9870605, -2325.9230957, 1823.9427490, -4571.3920898, 4428.9101562
3: -1039.7465820, 2749.6359863, -899.5002441, 2358.9836426, -3398.7302246, 3649.1362305
4: -3023.0925293, 2047.6083984, -2562.2343750, 1776.0528564, -4799.1445312, 4609.8422852

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.9232770, upper bound: 3611.7862806
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.9232770, upper bound: 3611.7862582
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2308.8933105, 1978.9342041, -1943.8889160, 1682.8842773, -3991.7775879, 3922.8232422
1: -1859.4483643, 1938.1870117, -1560.3826904, 1649.3551025, -3508.8029785, 3498.5698242
2: -2747.4206543, 2102.9699707, -2288.9770508, 1787.2509766, -4534.6718750, 4391.9458008
3: -1039.7386475, 2749.6105957, -883.4014282, 2317.5876465, -3357.3261719, 3633.0119629
4: -3023.0610352, 2047.5914307, -2520.9567871, 1740.5045166, -4763.5649414, 4568.5483398

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.9232770, upper bound: 3611.9401320
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.9232770, upper bound: 3611.9362740
time: 5.20 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2463.8439941, 2115.7326660, -2159.6970215, 1876.9660645, -4340.8095703, 4275.4296875
1: -1983.9792480, 2072.1596680, -1732.6105957, 1840.3956299, -3824.3750000, 3804.7692871
2: -2928.6206055, 2246.4086914, -2538.4934082, 1992.0585938, -4920.6787109, 4784.9008789
3: -1107.6740723, 2935.6577148, -981.9328613, 2576.9663086, -3684.6398926, 3917.5900879
4: -3223.6059570, 2189.0800781, -2796.5915527, 1940.0974121, -5163.7031250, 4985.6718750

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0164505, upper bound: 3611.8425626
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0169921, upper bound: 3611.8425372
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2463.7961426, 2115.6982422, -2079.5249023, 1804.6406250, -4268.4360352, 4195.2231445
1: -1983.9399414, 2072.1262207, -1669.4020996, 1770.1657715, -3754.1057129, 3741.5280762
2: -2928.5605469, 2246.3728027, -2447.5773926, 1916.9589844, -4845.5195312, 4693.9501953
3: -1107.6562500, 2935.6022949, -947.0042114, 2482.3625488, -3590.0185547, 3882.6062012
4: -3223.5397949, 2189.0444336, -2694.8488770, 1865.7957764, -5089.3344727, 4883.8935547

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0164505, upper bound: 3611.8425626
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0169921, upper bound: 3611.8425372
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -2374.0427246, 2035.3471680, -2159.6970215, 1876.9660645, -4251.0083008, 4195.0439453
1: -1912.6689453, 1994.0125732, -1732.6105957, 1840.3956299, -3753.0644531, 3726.6225586
2: -2825.5732422, 2163.0234375, -2538.4934082, 1992.0585938, -4817.6313477, 4701.5161133
3: -1069.0504150, 2828.6435547, -981.9328613, 2576.9663086, -3646.0166016, 3810.5759277
4: -3108.4506836, 2106.2163086, -2796.5915527, 1940.0974121, -5048.5478516, 4902.8076172

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.9232603, upper bound: 3611.8425514
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.9232603, upper bound: 3611.8425372
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2374.0041504, 2035.3198242, -2079.5249023, 1804.6406250, -4178.6445312, 4114.8447266
1: -1912.6373291, 1993.9860840, -1669.4020996, 1770.1657715, -3682.8032227, 3663.3876953
2: -2825.5244141, 2162.9956055, -2447.5773926, 1916.9589844, -4742.4829102, 4610.5732422
3: -1069.0361328, 2828.5988770, -947.0042114, 2482.3625488, -3551.3986816, 3775.6030273
4: -3108.3972168, 2106.1884766, -2694.8488770, 1865.7957764, -4974.1923828, 4801.0371094

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.9232603, upper bound: 3611.8425598
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.9232603, upper bound: 3611.8425372
time: 0.94 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_A2_A2

### Backsubstitution after applying NS history:
0: -1974.8409424, 1716.9732666, -4018.4333496, 3624.4799805, -5428.0859375, 5735.4067383
1: -1591.1787109, 1686.4777832, -3240.0883789, 3568.2119141, -5016.4169922, 4926.5649414
2: -2367.1381836, 1823.9008789, -4888.9609375, 3836.1330566, -6048.7060547, 6692.7216797
3: -892.7864990, 2385.7099609, -1799.2294922, 4929.4150391, -5771.2709961, 4145.9760742
4: -2601.9707031, 1778.6447754, -5347.5009766, 3726.8911133, -6165.6118164, 7116.9780273

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A1_A2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0235804, upper bound: 3612.8608157
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_A2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0237672, upper bound: 3612.9616227
time: 0.76 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 2.27 seconds
NS_A1_B2_B2_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.7403670, upper bound: 3613.4711716
NS_A1_B2_B2_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.7403670, upper bound: 3613.4735986
NS_A1_B2_B2_B1_B1_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.9132242, upper bound: 3611.3824673
NS_A1_B2_B2_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.9150142, upper bound: 3613.4734865
NS_A1_B2_B2_B1_B1_A2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.9132242, upper bound: 3611.3831078
NS_A1_B2_B2_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.9150142, upper bound: 3613.4737078
NS_A1_B2_B2_B1_B2_A1_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.6749365, upper bound: 3611.3933792
NS_A1_B2_B2_B1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.6769015, upper bound: 3613.5145017
NS_A1_B2_B2_B1_B2_A1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.6749365, upper bound: 3611.3933880
NS_A1_B2_B2_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.6769015, upper bound: 3613.5145458
NS_A1_B2_B2_B1_B2_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.9132261, upper bound: 3611.3933419
NS_A1_B2_B2_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.9152004, upper bound: 3613.5143632
NS_A1_B2_B2_B1_B2_A2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.9132261, upper bound: 3611.3933442
NS_A1_B2_B2_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.9152004, upper bound: 3613.5143632
NS_A1_B2_B2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.7862835, upper bound: 3614.0164730
NS_A1_B2_B2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.7862582, upper bound: 3614.0170017
NS_A1_B2_B2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.7862835, upper bound: 3614.0164681
NS_A1_B2_B2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.7862582, upper bound: 3614.0169967
NS_A1_B2_B2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.7862806, upper bound: 3613.9232770
NS_A1_B2_B2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.7862582, upper bound: 3613.9232770
NS_A1_B2_B2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.7862806, upper bound: 3613.9244825
NS_A1_B2_B2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.7862582, upper bound: 3613.9244825
NS_A1_B2_B2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.8425626, upper bound: 3614.0164505
NS_A1_B2_B2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.8425372, upper bound: 3614.0169971
NS_A1_B2_B2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.8425626, upper bound: 3614.0164505
NS_A1_B2_B2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.8425372, upper bound: 3614.0169921
NS_A1_B2_B2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.8425598, upper bound: 3613.9232603
NS_A1_B2_B2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.8425372, upper bound: 3613.9232603
NS_A1_B2_B2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.8425598, upper bound: 3613.9238978
NS_A1_B2_B2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.8425372, upper bound: 3613.9238978
NS_A2_B1_A1_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3613.5554447, upper bound: 3612.5875344
NS_A2_B1_A1_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3613.5554447, upper bound: 3612.5936792
NS_A2_B1_A1_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3613.7943513, upper bound: 3612.5661049
NS_A2_B1_A1_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3614.0217443, upper bound: 3612.5662630
NS_A2_B1_A1_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3613.5319262, upper bound: 3612.5790557
NS_A2_B1_A1_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3613.5319262, upper bound: 3612.5414763
NS_A2_B1_A1_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3613.7944839, upper bound: 3612.5661049
NS_A2_B1_A1_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3614.0223578, upper bound: 3612.5662630
NS_A2_B1_A2_A1_A1_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.3824673, upper bound: 3611.6749334
NS_A2_B1_A2_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3613.4735125, upper bound: 3611.6767121
NS_A2_B1_A2_A1_A1_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.3824673, upper bound: 3611.7581381
NS_A2_B1_A2_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3613.4735125, upper bound: 3611.7599242
NS_A2_B1_A2_A1_A1_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.3824673, upper bound: 3611.9132242
NS_A2_B1_A2_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3613.4734865, upper bound: 3611.9150142
NS_A2_B1_A2_A1_A1_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.3824673, upper bound: 3612.7374182
NS_A2_B1_A2_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3613.4734865, upper bound: 3612.7354297
NS_A2_B1_A2_A1_A2_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.3933792, upper bound: 3611.6749365
NS_A2_B1_A2_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3613.5144961, upper bound: 3611.6769015
NS_A2_B1_A2_A1_A2_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.3933792, upper bound: 3611.7581421
NS_A2_B1_A2_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3613.5144961, upper bound: 3611.7601164
NS_A2_B1_A2_A1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.3933793, upper bound: 3611.9131581
NS_A2_B1_A2_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3613.5144836, upper bound: 3611.9152004
NS_A2_B1_A2_A1_A2_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -3611.3933793, upper bound: 3612.5749284
NS_A2_B1_A2_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3613.5144836, upper bound: 3612.6272545
NS_A2_B1_A2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3614.0164681, upper bound: 3611.7862835
NS_A2_B1_A2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3614.0169967, upper bound: 3611.7862582
NS_A2_B1_A2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3614.0164681, upper bound: 3611.8926687
NS_A2_B1_A2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3614.0169967, upper bound: 3611.8927471
NS_A2_B1_A2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3613.9232770, upper bound: 3611.7862806
NS_A2_B1_A2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3613.9232770, upper bound: 3611.7862582
NS_A2_B1_A2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3613.9232770, upper bound: 3611.9401320
NS_A2_B1_A2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3613.9232770, upper bound: 3611.9362740
NS_A2_B1_A2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3614.0164505, upper bound: 3611.8425626
NS_A2_B1_A2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3614.0169921, upper bound: 3611.8425372
NS_A2_B1_A2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3614.0164505, upper bound: 3611.8425626
NS_A2_B1_A2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3614.0169921, upper bound: 3611.8425372
NS_A2_B1_A2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3613.9232603, upper bound: 3611.8425514
NS_A2_B1_A2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3613.9232603, upper bound: 3611.8425372
NS_A2_B1_A2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3613.9232603, upper bound: 3611.8425598
NS_A2_B1_A2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.27
Output dim: 0, lower bound: -3613.9232603, upper bound: 3611.8425372
NS_A2_B2_A1_B2_A1_A2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -3613.0235804, upper bound: 3612.8608157
NS_A2_B2_A1_B2_A1_A2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.27
Output dim: 0, lower bound: -3613.0237672, upper bound: 3612.9616227

## BFS NS instance: NS_A1_B2_B2_B1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2087.0527344, 1821.6109619, -2473.9282227, 2141.3574219, -4228.4086914, 4295.5385742
1: -1676.0616455, 1787.3134766, -1992.6701660, 2097.6123047, -3773.6738281, 3779.9836426
2: -2458.9858398, 1935.0238037, -2944.6721191, 2274.1499023, -4733.1342773, 4879.6958008
3: -952.0138550, 2498.1445312, -1116.6373291, 2955.8496094, -3907.8632812, 3614.7817383
4: -2706.1860352, 1884.8574219, -3239.4680176, 2219.0222168, -4925.2080078, 5124.3251953

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 43

## BFS NS instance: NS_A1_B2_B2_B1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2078.6567383, 1813.3693848, -2474.1982422, 2141.5588379, -4220.2158203, 4287.5673828
1: -1669.1791992, 1779.3377686, -1992.8914795, 2097.8081055, -3766.9870605, 3772.2285156
2: -2448.2922363, 1926.3680420, -2945.0190430, 2274.3603516, -4722.6513672, 4871.3872070
3: -948.7012329, 2486.9326172, -1116.7402344, 2956.1643066, -3904.8654785, 3603.6723633
4: -2694.5634766, 1875.8934326, -3239.8461914, 2219.2329102, -4913.7963867, 5115.7397461

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 43

## BFS NS instance: NS_A1_B2_B2_B1_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -2198.3762207, 1923.3669434, -2474.1174316, 2141.5014648, -4339.8779297, 4397.4843750
1: -1764.6110840, 1887.4392090, -1992.8253174, 2097.7521973, -3862.3630371, 3880.2644043
2: -2587.5109863, 2042.6828613, -2944.9138184, 2274.3007812, -4861.8115234, 4987.5966797
3: -1003.2032471, 2632.3742676, -1116.7115479, 2956.0703125, -3959.2734375, 3749.0856934
4: -2848.5363770, 1990.8115234, -3239.7314453, 2219.1733398, -5067.7094727, 5230.5429688

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_B2_B1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_B2_B1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_B1_B1_A2_A1_B2_A1

### Relational analysis result of NS_A1_B2_B2_B1_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.5655836, upper bound: 3613.4711846
time: 0.68 seconds

## Relational analysis of NS_A1_B2_B2_B1_B1_A2_A1_B2_A2

### Relational analysis result of NS_A1_B2_B2_B1_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.5655836, upper bound: 3613.4734928
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_B2_B1_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -2119.1289062, 1843.8908691, -2474.1389160, 2141.5170898, -4260.6459961, 4318.0292969
1: -1701.5185547, 1809.7534180, -1992.8433838, 2097.7673340, -3799.2858887, 3802.5966797
2: -2495.1850586, 1959.6323242, -2944.9438477, 2274.3171387, -4769.5019531, 4904.5761719
3: -966.2225342, 2534.0144043, -1116.7192383, 2956.0961914, -3922.3188477, 3650.7331543
4: -2746.3649902, 1907.2337646, -3239.7634277, 2219.1901855, -4965.5551758, 5146.9970703

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_B2_B1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B2_B1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_B2_B1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_B1_B1_A2_A2_B2_A1

### Relational analysis result of NS_A1_B2_B2_B1_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.4485931, upper bound: 3613.4711532
time: 0.71 seconds

## Relational analysis of NS_A1_B2_B2_B1_B1_A2_A2_B2_A2

### Relational analysis result of NS_A1_B2_B2_B1_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.4485931, upper bound: 3613.4737141
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_B2_B1_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -2174.8547363, 1907.8500977, -2491.0051270, 2151.0444336, -4325.8994141, 4398.8554688
1: -1745.8293457, 1872.0988770, -2007.2673340, 2106.8044434, -3852.6333008, 3879.3662109
2: -2560.6362305, 2026.8896484, -2966.1025391, 2284.0268555, -4844.6621094, 4992.9921875
3: -993.0927124, 2605.6098633, -1123.8806152, 2976.8115234, -3969.9042969, 3729.4899902
4: -2818.8544922, 1974.6855469, -3262.0461426, 2228.5676270, -5047.4218750, 5236.7314453

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_B2_B1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_B2_B1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_B1_B2_A1_A1_B2_A1

### Relational analysis result of NS_A1_B2_B2_B1_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.5066817, upper bound: 3613.5041910
time: 0.75 seconds

## Relational analysis of NS_A1_B2_B2_B1_B2_A1_A1_B2_A2

### Relational analysis result of NS_A1_B2_B2_B1_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.5066817, upper bound: 3613.5145017
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_B2_B1_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -2089.4301758, 1822.5927734, -2491.0263672, 2151.0600586, -4240.4902344, 4313.6191406
1: -1677.8387451, 1788.5040283, -2007.2845459, 2106.8195801, -3784.6582031, 3795.7880859
2: -2461.1071777, 1936.3327637, -2966.1315918, 2284.0424805, -4745.1489258, 4902.4643555
3: -953.5634155, 2499.8732910, -1123.8878174, 2976.8361816, -3930.3996582, 3623.7612305
4: -2708.5678711, 1885.4360352, -3262.0776367, 2228.5837402, -4937.1513672, 5147.5131836

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_B2_B1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B2_B1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_B2_B1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_B1_B2_A1_A2_B2_A1

### Relational analysis result of NS_A1_B2_B2_B1_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.7405594, upper bound: 3613.5118350
time: 0.85 seconds

## Relational analysis of NS_A1_B2_B2_B1_B2_A1_A2_B2_A2

### Relational analysis result of NS_A1_B2_B2_B1_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.7405594, upper bound: 3613.5145458
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_B2_B1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -2198.3762207, 1923.3669434, -2490.9343262, 2150.9934082, -4349.3696289, 4414.3012695
1: -1764.6110840, 1887.4392090, -2007.2089844, 2106.7548828, -3871.3659668, 3894.6479492
2: -2587.5109863, 2042.6828613, -2966.0122070, 2283.9729004, -4871.4838867, 5008.6928711
3: -1003.2032471, 2632.3742676, -1123.8543701, 2976.7285156, -3979.9316406, 3756.2285156
4: -2848.5363770, 1990.8115234, -3261.9475098, 2228.5136719, -5077.0498047, 5252.7587891

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_B2_B1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_B2_B1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_B1_B2_A2_A1_B2_A1

### Relational analysis result of NS_A1_B2_B2_B1_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.5656478, upper bound: 3613.5121225
time: 0.74 seconds

## Relational analysis of NS_A1_B2_B2_B1_B2_A2_A1_B2_A2

### Relational analysis result of NS_A1_B2_B2_B1_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.5656478, upper bound: 3613.5143689
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_B2_B1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -2119.1289062, 1843.8908691, -2490.9558105, 2151.0092773, -4270.1381836, 4334.8461914
1: -1701.5185547, 1809.7534180, -2007.2266846, 2106.7702637, -3808.2885742, 3816.9799805
2: -2495.1850586, 1959.6323242, -2966.0415039, 2283.9897461, -4779.1748047, 4925.6738281
3: -966.2225342, 2534.0144043, -1123.8619385, 2976.7546387, -3942.9770508, 3657.8762207
4: -2746.3649902, 1907.2337646, -3261.9794922, 2228.5305176, -4974.8955078, 5169.2133789

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_B2_B1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B2_B1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_B2_B1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_B1_B2_A2_A2_B2_A1

### Relational analysis result of NS_A1_B2_B2_B1_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.4478959, upper bound: 3613.5122688
time: 0.69 seconds

## Relational analysis of NS_A1_B2_B2_B1_B2_A2_A2_B2_A2

### Relational analysis result of NS_A1_B2_B2_B1_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.4478959, upper bound: 3613.5146120
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1898.1889648, 1651.9700928, -2150.3303223, 1842.0078125, -3740.1965332, 3802.3002930
1: -1523.0819092, 1619.5979004, -1732.0383301, 1801.7005615, -3324.7824707, 3351.6362305
2: -2233.1958008, 1754.3031006, -2560.4812012, 1955.0076904, -4188.2031250, 4314.7841797
3: -864.3604736, 2266.1057129, -964.0993652, 2554.5534668, -3418.9138184, 3230.2050781
4: -2459.5205078, 1708.4790039, -2817.6867676, 1905.2983398, -4364.8183594, 4526.1660156

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_B2_B2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B2_B2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_B2_B2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_B2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_B2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_B2_B2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_B2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_B2_B2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B2_B2_A1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B2_B2_B2_A1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.8518215, upper bound: 3612.0542885
time: 0.62 seconds

## Relational analysis of NS_A1_B2_B2_B2_A1_B1_A1_B1_B2

### Relational analysis result of NS_A1_B2_B2_B2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.9472110, upper bound: 3614.0217299
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1958.3607178, 1701.9300537, -2369.2851562, 2033.5694580, -3991.9299316, 4071.2153320
1: -1570.9538574, 1668.1325684, -1907.3885498, 1990.6264648, -3561.5803223, 3575.5209961
2: -2303.4624023, 1806.8338623, -2816.4323730, 2158.6179199, -4462.0800781, 4623.2661133
3: -890.7656250, 2336.5083008, -1064.2408447, 2821.5429688, -3712.3083496, 3400.7490234
4: -2537.5786133, 1759.7368164, -3100.7326660, 2104.0622559, -4641.6396484, 4860.4697266

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_B2_B2_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B2_B2_B2_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.9456755, upper bound: 3613.5151710
time: 0.87 seconds

## Relational analysis of NS_A1_B2_B2_B2_A1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2_B2_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.9471559, upper bound: 3614.0224362
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1886.1367188, 1634.8516846, -2150.3022461, 1841.9869385, -3728.1232910, 3785.1538086
1: -1514.3289795, 1602.4230957, -1732.0153809, 1801.6802979, -3316.0092773, 3334.4384766
2: -2221.6240234, 1736.5588379, -2560.4448242, 1954.9857178, -4176.6098633, 4297.0019531
3: -857.4950562, 2249.8559570, -964.0885620, 2554.5202637, -3412.0153809, 3213.9445801
4: -2446.4858398, 1691.0688477, -2817.6464844, 1905.2764893, -4351.7622070, 4508.7153320

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_B2_B2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B2_B2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_B2_B2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_B2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8926655, upper bound: 3614.0057492
time: 0.82 seconds

## Relational analysis of NS_A1_B2_B2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_B2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8926655, upper bound: 3614.0164681
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1911.0676270, 1655.3215332, -2369.2536621, 2033.5465088, -3944.6137695, 4024.5751953
1: -1534.1257324, 1622.5676270, -1907.3627930, 1990.6043701, -3524.7299805, 3529.9304199
2: -2250.5156250, 1758.2691650, -2816.3923340, 2158.5939941, -4409.1093750, 4574.6606445
3: -868.6502686, 2278.7956543, -1064.2290039, 2821.5063477, -3690.1562500, 3343.0244141
4: -2478.4641113, 1712.2841797, -3100.6887207, 2104.0380859, -4582.5014648, 4812.9726562

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_B2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8927447, upper bound: 3614.0057432
time: 0.77 seconds

## Relational analysis of NS_A1_B2_B2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_B2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8927440, upper bound: 3614.0169967
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1898.1889648, 1651.9700928, -2112.9846191, 1807.5080566, -3705.6970215, 3764.9545898
1: -1523.0819092, 1619.5979004, -1702.4290771, 1769.0079346, -3292.0898438, 3322.0268555
2: -2233.1958008, 1754.3031006, -2517.6420898, 1920.4938965, -4153.6894531, 4271.9453125
3: -864.3604736, 2266.1057129, -948.3652344, 2511.1191406, -3375.4794922, 3214.4707031
4: -2459.5205078, 1708.4790039, -2768.2109375, 1869.7274170, -4329.2470703, 4476.6899414

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_A1_B1_B1

### Relational analysis result of NS_A1_B2_B2_B2_A1_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.8351743, upper bound: 3613.1177574
time: 0.67 seconds

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_A1_B1_B2

### Relational analysis result of NS_A1_B2_B2_B2_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8351743, upper bound: 3613.9232770
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1958.3607178, 1701.9300537, -2267.7407227, 1943.0462646, -3901.4069824, 3969.6708984
1: -1570.9538574, 1668.1325684, -1826.4947510, 1902.9366455, -3473.8901367, 3494.6271973
2: -2303.4624023, 1806.8338623, -2699.4497070, 2064.6474609, -4368.1098633, 4506.2836914
3: -890.7656250, 2336.5083008, -1020.9647827, 2700.8718262, -3591.6374512, 3357.4731445
4: -2537.5786133, 1759.7368164, -2970.3144531, 2010.5484619, -4548.1264648, 4730.0512695

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_A1_B2_B1

### Relational analysis result of NS_A1_B2_B2_B2_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8897573, upper bound: 3613.7739816
time: 0.77 seconds

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2_B2_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8897573, upper bound: 3613.9232830
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1886.1367188, 1634.8516846, -2112.9633789, 1807.4929199, -3693.6293945, 3747.8149414
1: -1514.3289795, 1602.4230957, -1702.4114990, 1768.9930420, -3283.3220215, 3304.8344727
2: -2221.6240234, 1736.5588379, -2517.6135254, 1920.4781494, -4142.1020508, 4254.1718750
3: -857.4950562, 2249.8559570, -948.3577271, 2511.0942383, -3368.5893555, 3198.2136230
4: -2446.4858398, 1691.0688477, -2768.1801758, 1869.7113037, -4316.1972656, 4459.2490234

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_B2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.9406430, upper bound: 3613.9244276
time: 0.72 seconds

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_B2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.9406422, upper bound: 3613.9244825
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1911.0676270, 1655.3215332, -2267.7163086, 1943.0289307, -3854.0964355, 3923.0378418
1: -1534.1257324, 1622.5676270, -1826.4748535, 1902.9197998, -3437.0454102, 3449.0424805
2: -2250.5156250, 1758.2691650, -2699.4179688, 2064.6298828, -4315.1455078, 4457.6870117
3: -868.6502686, 2278.7956543, -1020.9561768, 2700.8439941, -3569.4941406, 3299.7517090
4: -2478.4641113, 1712.2841797, -2970.2800293, 2010.5302734, -4488.9941406, 4682.5639648

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_B2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.9406422, upper bound: 3613.9244337
time: 0.90 seconds

## Relational analysis of NS_A1_B2_B2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_B2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.9406422, upper bound: 3613.9244825
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -2057.4877930, 1791.9794922, -2182.4965820, 1870.5206299, -3928.0083008, 3974.4760742
1: -1651.0880127, 1757.4875488, -1757.9085693, 1830.2285156, -3481.3164062, 3515.3959961
2: -2419.5642090, 1902.2930908, -2598.9436035, 1985.5732422, -4405.1376953, 4501.2368164
3: -936.3309937, 2457.2944336, -979.1383667, 2594.6718750, -3531.0029297, 3436.4328613
4: -2664.9460449, 1852.7579346, -2859.6596680, 1935.0572510, -4600.0029297, 4712.4169922

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_B2_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B2_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_B2_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_B2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.5955662, upper bound: 3614.0215804
time: 0.62 seconds

## Relational analysis of NS_A1_B2_B2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_B2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.5955662, upper bound: 3614.0217802
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -2147.6633301, 1866.9766846, -2435.3703613, 2091.4899902, -4239.1528320, 4302.3461914
1: -1723.0256348, 1830.5748291, -1961.2313232, 2048.3386230, -3771.3642578, 3791.8061523
2: -2524.4301758, 1981.3300781, -2895.4282227, 2220.5783691, -4745.0073242, 4876.7583008
3: -976.4069214, 2562.8596191, -1094.4318848, 2902.0563965, -3878.4633789, 3657.2915039
4: -2781.0791016, 1929.8286133, -3187.0344238, 2164.1337891, -4945.2119141, 5116.8632812

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_B2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.5955662, upper bound: 3614.0220227
time: 0.79 seconds

## Relational analysis of NS_A1_B2_B2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_B2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.5955662, upper bound: 3614.0224361
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -2004.5377197, 1741.3880615, -2182.4545898, 1870.4897461, -3875.0273438, 3923.8420410
1: -1609.5473633, 1708.5133057, -1757.8740234, 1830.1987305, -3439.7456055, 3466.3872070
2: -2360.2414551, 1850.3981934, -2598.8906250, 1985.5411377, -4345.7827148, 4449.2890625
3: -912.7960815, 2393.7290039, -979.1221924, 2594.6228027, -3507.4187012, 3372.8510742
4: -2598.2106934, 1800.9992676, -2859.6018066, 1935.0251465, -4533.2358398, 4660.6000977

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_B2_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B2_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_B2_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_B2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8425372, upper bound: 3614.0164097
time: 0.74 seconds

## Relational analysis of NS_A1_B2_B2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_B2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8425372, upper bound: 3614.0164555
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2059.3156738, 1787.4958496, -2435.3212891, 2091.4550781, -4150.7705078, 4222.8173828
1: -1653.2061768, 1753.3509521, -1961.1916504, 2048.3049316, -3701.5112305, 3714.5424805
2: -2423.8579102, 1898.7449951, -2895.3671875, 2220.5422363, -4644.3999023, 4794.1123047
3: -937.8716431, 2458.5400391, -1094.4136963, 2902.0002441, -3839.8718262, 3552.9533691
4: -2668.7690430, 1848.1281738, -3186.9680176, 2164.0974121, -4832.8657227, 5035.0961914

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_B2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8425372, upper bound: 3614.0169532
time: 0.73 seconds

## Relational analysis of NS_A1_B2_B2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_B2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8425372, upper bound: 3614.0169971
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2057.4877930, 1791.9794922, -2146.8740234, 1838.0313721, -3895.5187988, 3938.8535156
1: -1651.0880127, 1757.4875488, -1729.7518311, 1799.1739502, -3450.2619629, 3487.2390137
2: -2419.5642090, 1902.2930908, -2558.7517090, 1952.8004150, -4372.3647461, 4461.0444336
3: -936.3309937, 2457.2944336, -964.2346191, 2553.4541016, -3489.7846680, 3421.5288086
4: -2664.9460449, 1852.7579346, -2813.3908691, 1901.5115967, -4566.4575195, 4666.1484375

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_B2_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_B2_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_B2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.2025809, upper bound: 3613.9232158
time: 0.80 seconds

## Relational analysis of NS_A1_B2_B2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_B2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.2025809, upper bound: 3613.9232603
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2147.6633301, 1866.9766846, -2330.8544922, 1997.7227783, -4145.3857422, 4197.8310547
1: -1723.0256348, 1830.5748291, -1878.0645752, 1956.9641113, -3679.9897461, 3708.6394043
2: -2524.4301758, 1981.3300781, -2775.2331543, 2122.7390137, -4647.1689453, 4756.5629883
3: -976.4069214, 2562.8596191, -1049.2624512, 2777.5441895, -3753.9511719, 3612.1218262
4: -2781.0791016, 1929.8286133, -3053.0541992, 2067.6601562, -4848.7392578, 4982.8823242

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_B2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.2046004, upper bound: 3613.9232158
time: 0.76 seconds

## Relational analysis of NS_A1_B2_B2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_B2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.2046004, upper bound: 3613.9232603
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -2004.5377197, 1741.3880615, -2146.8388672, 1838.0063477, -3842.5439453, 3888.2263184
1: -1609.5473633, 1708.5133057, -1729.7224121, 1799.1496582, -3408.6970215, 3438.2358398
2: -2360.2414551, 1850.3981934, -2558.7065430, 1952.7745361, -4313.0161133, 4409.1044922
3: -912.7960815, 2393.7290039, -964.2218018, 2553.4125977, -3466.2084961, 3357.9506836
4: -2598.2106934, 1800.9992676, -2813.3415527, 1901.4855957, -4499.6962891, 4614.3408203

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_B2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_B2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_B2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8425372, upper bound: 3613.9239038
time: 0.64 seconds

## Relational analysis of NS_A1_B2_B2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_B2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8425372, upper bound: 3613.9238978
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2059.3156738, 1787.4958496, -2330.8149414, 1997.6945801, -4057.0102539, 4118.3105469
1: -1653.2061768, 1753.3509521, -1878.0319824, 1956.9366455, -3610.1428223, 3631.3823242
2: -2423.8579102, 1898.7449951, -2775.1828613, 2122.7094727, -4546.5673828, 4673.9277344
3: -937.8716431, 2458.5400391, -1049.2478027, 2777.4987793, -3715.3703613, 3507.7878418
4: -2668.7690430, 1848.1281738, -3052.9995117, 2067.6306152, -4736.3989258, 4901.1269531

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_B2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_B2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8425372, upper bound: 3613.9238978
time: 0.63 seconds

## Relational analysis of NS_A1_B2_B2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_B2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8425372, upper bound: 3613.9239038
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2246.4812012, 1929.5992432, -1806.4156494, 1566.7838135, -3813.2648926, 3736.0146484
1: -1809.8557129, 1890.0133057, -1448.1267090, 1535.7125244, -3345.5683594, 3338.1401367
2: -2674.0568848, 2050.3576660, -2121.2690430, 1665.1363525, -4339.1923828, 4171.6269531
3: -1010.8537598, 2677.3208008, -821.0072021, 2149.0517578, -3159.9047852, 3498.3276367
4: -2941.8566895, 1997.8632812, -2336.1884766, 1619.7669678, -4561.6235352, 4334.0512695

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 43

## BFS NS instance: NS_A2_B1_A1_A2_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2246.5556641, 1929.6558838, -1876.1182861, 1631.0578613, -3877.6135254, 3805.7741699
1: -1809.9165039, 1890.0681152, -1505.1042480, 1599.9382324, -3409.8547363, 3395.1723633
2: -2674.1523438, 2050.4167480, -2207.2058105, 1733.9434814, -4408.0957031, 4257.6225586
3: -1010.8828735, 2677.4091797, -854.4660034, 2237.2507324, -3248.1335449, 3531.8752441
4: -2941.9609375, 1997.9218750, -2429.7241211, 1686.2048340, -4628.1655273, 4427.6459961

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 43

## BFS NS instance: NS_A2_B1_A1_A2_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -2275.2109375, 1951.2884521, -1870.3756104, 1626.7907715, -3902.0017090, 3821.6635742
1: -1834.3237305, 1910.5051270, -1500.6270752, 1595.8714600, -3430.1948242, 3411.1318359
2: -2711.9321289, 2072.6755371, -2201.0100098, 1729.5772705, -4441.5092773, 4273.6855469
3: -1023.4332886, 2714.7602539, -852.2728271, 2231.0637207, -3254.4970703, 3567.0332031
4: -2981.8933105, 2020.3684082, -2422.7180176, 1681.8466797, -4663.7392578, 4443.0859375

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A2_A1_B1

### Relational analysis result of NS_A2_B1_A1_A2_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7943225, upper bound: 3612.5661049
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A2_A1_B2

### Relational analysis result of NS_A2_B1_A1_A2_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7943225, upper bound: 3612.5661049
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -2265.5356445, 1944.4228516, -1874.7276611, 1630.3983154, -3895.9338379, 3819.1503906
1: -1826.0900879, 1903.8232422, -1504.1112061, 1599.4117432, -3425.5019531, 3407.9335938
2: -2698.0419922, 2065.4792480, -2206.0766602, 1733.4128418, -4431.4550781, 4271.5556641
3: -1019.4014893, 2701.7983398, -854.1786499, 2236.1303711, -3255.5317383, 3555.9770508
4: -2967.3142090, 2012.9240723, -2428.2998047, 1685.5743408, -4652.8881836, 4441.2236328

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A2_A2_B1

### Relational analysis result of NS_A2_B1_A1_A2_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0217199, upper bound: 3612.5662630
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A2_A2_B2

### Relational analysis result of NS_A2_B1_A1_A2_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0217199, upper bound: 3612.5662630
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -2025.1978760, 1735.7535400, -2088.8212891, 1820.4105225, -3845.6079102, 3824.5744629
1: -1630.8597412, 1698.8618164, -1676.7429199, 1785.9445801, -3416.8037109, 3375.6044922
2: -2414.0693359, 1843.9260254, -2458.2387695, 1933.5567627, -4347.6259766, 4302.1650391
3: -909.3659668, 2411.3422852, -952.1087646, 2497.4025879, -3406.7685547, 3363.4509277
4: -2656.7705078, 1798.1285400, -2706.6801758, 1882.7955322, -4539.5659180, 4504.8085938

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A1_A1_B1

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5305352, upper bound: 3612.5714756
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A1_A1_B2

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5305352, upper bound: 3612.5790544
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -2220.6953125, 1907.8693848, -2088.8212891, 1820.4105225, -4041.1049805, 3996.6906738
1: -1789.3013916, 1868.8481445, -1676.7429199, 1785.9445801, -3575.2460938, 3545.5908203
2: -2644.3540039, 2027.4615479, -2458.2387695, 1933.5567627, -4577.9106445, 4485.7001953
3: -999.4689331, 2647.6333008, -952.1087646, 2497.4025879, -3496.8710938, 3599.7419434
4: -2909.1926270, 1975.2839355, -2706.6801758, 1882.7955322, -4791.9873047, 4681.9633789

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A1_A2_B1

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5305352, upper bound: 3612.5243954
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A1_A2_B2

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5305352, upper bound: 3612.5414763
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -2275.1745605, 1951.2568359, -2071.7307129, 1806.2282715, -4081.4025879, 4022.9875488
1: -1834.2916260, 1910.4754639, -1663.0727539, 1772.0727539, -3606.3637695, 3573.5480957
2: -2711.8693848, 2072.6433105, -2438.4123535, 1918.5635986, -4630.4321289, 4511.0556641
3: -1023.4215698, 2714.7097168, -944.5802002, 2477.5134277, -3500.9350586, 3659.2897949
4: -2981.8266602, 2020.3344727, -2684.7321777, 1868.1944580, -4850.0209961, 4705.0664062

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A2_A1_A1

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.7783137, upper bound: 3612.5657832
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A2_A1_A2

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.7783137, upper bound: 3612.5661036
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -2265.4951172, 1944.3872070, -2077.6520996, 1811.1444092, -4076.6396484, 4022.0388184
1: -1826.0545654, 1903.7893066, -1667.8284912, 1776.8822021, -3602.9360352, 3571.6176758
2: -2697.9753418, 2065.4416504, -2445.3176270, 1923.7659912, -4621.7412109, 4510.7592773
3: -1019.3862915, 2701.7453613, -947.1780396, 2484.4365234, -3503.8220215, 3648.9233398
4: -2967.2414551, 2012.8851318, -2692.3569336, 1873.2725830, -4840.5126953, 4705.2421875

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A2_A2_A1

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6160654, upper bound: 3612.5659917
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A2_A2_A2

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6160654, upper bound: 3612.5662567
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -2474.1948242, 2141.5566406, -2174.8547363, 1907.8500977, -4382.0439453, 4316.4111328
1: -1992.8879395, 2097.8056641, -1745.8293457, 1872.0988770, -3864.9863281, 3843.6347656
2: -2945.0112305, 2274.3581543, -2560.6362305, 2026.8896484, -4971.9008789, 4834.9931641
3: -1116.7397461, 2956.1594238, -993.0927124, 2605.6098633, -3722.3493652, 3949.2521973
4: -3239.8381348, 2219.2307129, -2818.8544922, 1974.6855469, -5214.5234375, 5038.0849609

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_A1_A1_B1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_A1_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4711872, upper bound: 3611.5064895
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_A1_A1_B1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_A1_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4711872, upper bound: 3611.6767121
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -2474.2155762, 2141.5717773, -2089.4301758, 1822.5927734, -4296.8081055, 4231.0019531
1: -1992.9056396, 2097.8203125, -1677.8387451, 1788.5040283, -3781.4089355, 3775.6591797
2: -2945.0402832, 2274.3737793, -2461.1071777, 1936.3327637, -4881.3730469, 4735.4794922
3: -1116.7470703, 2956.1850586, -953.5634155, 2499.8732910, -3616.6201172, 3909.7485352
4: -3239.8698730, 2219.2465820, -2708.5678711, 1885.4360352, -5125.3046875, 4927.8144531

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_A1_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_A1_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_A1_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_A1_A1_B1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_A1_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4711726, upper bound: 3611.7403670
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A2_A1_A1_B1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_A1_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4711726, upper bound: 3611.7599242
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -2474.1174316, 2141.5014648, -2198.3762207, 1923.3669434, -4397.4843750, 4339.8779297
1: -1992.8253174, 2097.7521973, -1764.6110840, 1887.4392090, -3880.2644043, 3862.3630371
2: -2944.9138184, 2274.3007812, -2587.5109863, 2042.6828613, -4987.5961914, 4861.8115234
3: -1116.7115479, 2956.0703125, -1003.2032471, 2632.3742676, -3749.0856934, 3959.2734375
4: -3239.7314453, 2219.1733398, -2848.5363770, 1990.8115234, -5230.5429688, 5067.7094727

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_A1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_A1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_A1_A1_B2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_A1_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4711793, upper bound: 3611.5655836
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A2_A1_A1_B2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_A1_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4711793, upper bound: 3611.9150142
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -2474.1389160, 2141.5170898, -2119.1289062, 1843.8908691, -4318.0297852, 4260.6459961
1: -1992.8433838, 2097.7673340, -1701.5185547, 1809.7534180, -3802.5966797, 3799.2858887
2: -2944.9438477, 2274.3171387, -2495.1850586, 1959.6323242, -4904.5761719, 4769.5019531
3: -1116.7192383, 2956.0961914, -966.2225342, 2534.0144043, -3650.7331543, 3922.3188477
4: -3239.7634277, 2219.1901855, -2746.3649902, 1907.2337646, -5146.9970703, 4965.5551758

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_A1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_A1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_A1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_A1_A1_B2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_A1_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4711542, upper bound: 3612.4485931
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_A1_A1_B2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_A1_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4711542, upper bound: 3612.7354297
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -2491.0051270, 2151.0444336, -2174.8547363, 1907.8500977, -4398.8554688, 4325.8994141
1: -2007.2673340, 2106.8044434, -1745.8293457, 1872.0988770, -3879.3659668, 3852.6333008
2: -2966.1025391, 2284.0268555, -2560.6362305, 2026.8896484, -4992.9921875, 4844.6621094
3: -1123.8806152, 2976.8115234, -993.0927124, 2605.6098633, -3729.4902344, 3969.9042969
4: -3262.0461426, 2228.5676270, -2818.8544922, 1974.6855469, -5236.7314453, 5047.4218750

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_A1_A2_B1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_A1_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5041921, upper bound: 3611.5066817
time: 0.98 seconds

## Relational analysis of NS_A2_B1_A2_A1_A2_B1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_A1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5041921, upper bound: 3611.6769015
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -2491.0263672, 2151.0600586, -2089.4301758, 1822.5927734, -4313.6191406, 4240.4902344
1: -2007.2845459, 2106.8195801, -1677.8387451, 1788.5040283, -3795.7880859, 3784.6582031
2: -2966.1315918, 2284.0424805, -2461.1071777, 1936.3327637, -4902.4643555, 4745.1489258
3: -1123.8878174, 2976.8361816, -953.5634155, 2499.8732910, -3623.7612305, 3930.3996582
4: -3262.0776367, 2228.5837402, -2708.5678711, 1885.4360352, -5147.5131836, 4937.1513672

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_A1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_A1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_A1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_A1_A2_B1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_A1_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5118350, upper bound: 3611.7405594
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A2_A1_A2_B1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_A1_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5118350, upper bound: 3611.7601164
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -2490.9311523, 2150.9909668, -2198.3762207, 1923.3669434, -4414.2978516, 4349.3671875
1: -2007.2062988, 2106.7521973, -1764.6110840, 1887.4392090, -3894.6452637, 3871.3632812
2: -2966.0080566, 2283.9707031, -2587.5109863, 2042.6828613, -5008.6884766, 4871.4814453
3: -1123.8531494, 2976.7253418, -1003.2032471, 2632.3742676, -3756.2270508, 3979.9287109
4: -3261.9431152, 2228.5114746, -2848.5363770, 1990.8115234, -5252.7548828, 5077.0478516

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_A1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5071075, upper bound: 3611.5657759
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_A1_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5071075, upper bound: 3611.9152004
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -2490.9519043, 2151.0070801, -2119.1289062, 1843.8908691, -4334.8422852, 4270.1357422
1: -2007.2238770, 2106.7675781, -1701.5185547, 1809.7534180, -3816.9772949, 3808.2861328
2: -2966.0373535, 2283.9870605, -2495.1850586, 1959.6323242, -4925.6699219, 4779.1718750
3: -1123.8604736, 2976.7507324, -966.2225342, 2534.0144043, -3657.8747559, 3942.9731445
4: -3261.9743652, 2228.5280762, -2746.3649902, 1907.2337646, -5169.2080078, 4974.8925781

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_A1_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5123020, upper bound: 3612.4396227
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_A1_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5123020, upper bound: 3612.6272545
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -2150.3303223, 1842.0078125, -1898.1889648, 1651.9700928, -3802.3002930, 3740.1962891
1: -1732.0383301, 1801.7005615, -1523.0819092, 1619.5979004, -3351.6362305, 3324.7824707
2: -2560.4812012, 1955.0074463, -2233.1958008, 1754.3031006, -4314.7841797, 4188.2031250
3: -964.0993042, 2554.5537109, -864.3604736, 2266.1057129, -3230.2050781, 3418.9140625
4: -2817.6867676, 1905.2982178, -2459.5205078, 1708.4790039, -4526.1660156, 4364.8178711

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B1_A1_A1

### Relational analysis result of NS_A2_B1_A2_A2_B1_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.0542885, upper bound: 3611.8518215
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B1_A1_A2

### Relational analysis result of NS_A2_B1_A2_A2_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0217299, upper bound: 3611.9472110
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2369.2851562, 2033.5694580, -1958.3607178, 1701.9300537, -4071.2153320, 3991.9299316
1: -1907.3885498, 1990.6264648, -1570.9538574, 1668.1325684, -3575.5209961, 3561.5803223
2: -2816.4323730, 2158.6179199, -2303.4624023, 1806.8338623, -4623.2661133, 4462.0800781
3: -1064.2408447, 2821.5429688, -890.7656250, 2336.5083008, -3400.7490234, 3712.3083496
4: -3100.7326660, 2104.0622559, -2537.5786133, 1759.7368164, -4860.4697266, 4641.6396484

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_A2_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5151710, upper bound: 3611.9456755
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0224315, upper bound: 3611.9471559
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2150.3022461, 1841.9869385, -1886.1367188, 1634.8516846, -3785.1538086, 3728.1230469
1: -1732.0153809, 1801.6802979, -1514.3289795, 1602.4230957, -3334.4384766, 3316.0092773
2: -2560.4448242, 1954.9857178, -2221.6240234, 1736.5588379, -4297.0019531, 4176.6093750
3: -964.0885620, 2554.5202637, -857.4950562, 2249.8559570, -3213.9445801, 3412.0153809
4: -2817.6464844, 1905.2764893, -2446.4858398, 1691.0688477, -4508.7153320, 4351.7622070

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0057432, upper bound: 3611.8926655
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0057432, upper bound: 3611.8926662
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2369.2536621, 2033.5465088, -1911.0676270, 1655.3215332, -4024.5751953, 3944.6137695
1: -1907.3627930, 1990.6043701, -1534.1257324, 1622.5676270, -3529.9304199, 3524.7299805
2: -2816.3923340, 2158.5939941, -2250.5156250, 1758.2691650, -4574.6606445, 4409.1093750
3: -1064.2290039, 2821.5063477, -868.6502686, 2278.7956543, -3343.0246582, 3690.1562500
4: -3100.6887207, 2104.0380859, -2478.4641113, 1712.2841797, -4812.9726562, 4582.5014648

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0057432, upper bound: 3611.8927447
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0057432, upper bound: 3611.8927440
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2112.9846191, 1807.5080566, -1898.1889648, 1651.9700928, -3764.9545898, 3705.6970215
1: -1702.4289551, 1769.0078125, -1523.0819092, 1619.5979004, -3322.0268555, 3292.0898438
2: -2517.6420898, 1920.4940186, -2233.1958008, 1754.3031006, -4271.9448242, 4153.6899414
3: -948.3652344, 2511.1188965, -864.3604736, 2266.1057129, -3214.4709473, 3375.4792480
4: -2768.2106934, 1869.7272949, -2459.5205078, 1708.4790039, -4476.6894531, 4329.2465820

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A2_A2_B1_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1177550, upper bound: 3611.8351743
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B1_A1_A2

### Relational analysis result of NS_A2_B1_A2_A2_B1_A2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1177550, upper bound: 3611.8351743
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2267.7407227, 1943.0462646, -1958.3607178, 1701.9300537, -3969.6708984, 3901.4069824
1: -1826.4947510, 1902.9366455, -1570.9538574, 1668.1325684, -3494.6274414, 3473.8903809
2: -2699.4497070, 2064.6474609, -2303.4624023, 1806.8338623, -4506.2836914, 4368.1098633
3: -1020.9647827, 2700.8718262, -890.7656250, 2336.5083008, -3357.4731445, 3591.6374512
4: -2970.3144531, 2010.5484619, -2537.5786133, 1759.7368164, -4730.0512695, 4548.1264648

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_A2_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7739814, upper bound: 3611.8897573
time: 0.89 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7739814, upper bound: 3611.8899841
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2112.9641113, 1807.4936523, -1886.1367188, 1634.8516846, -3747.8159180, 3693.6301270
1: -1702.4123535, 1768.9937744, -1514.3289795, 1602.4230957, -3304.8354492, 3283.3225098
2: -2517.6147461, 1920.4788818, -2221.6240234, 1736.5588379, -4254.1738281, 4142.1030273
3: -948.3579712, 2511.0954590, -857.4950562, 2249.8559570, -3198.2138672, 3368.5900879
4: -2768.1813965, 1869.7117920, -2446.4858398, 1691.0688477, -4459.2500000, 4316.1977539

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.9134761, upper bound: 3611.9362720
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.9134761, upper bound: 3611.9362720
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2267.7175293, 1943.0294189, -1911.0676270, 1655.3215332, -3923.0390625, 3854.0971680
1: -1826.4758301, 1902.9204102, -1534.1257324, 1622.5676270, -3449.0434570, 3437.0461426
2: -2699.4191895, 2064.6303711, -2250.5156250, 1758.2691650, -4457.6884766, 4315.1459961
3: -1020.9565430, 2700.8449707, -868.6502686, 2278.7956543, -3299.7521973, 3569.4951172
4: -2970.2817383, 2010.5307617, -2478.4641113, 1712.2841797, -4682.5659180, 4488.9946289

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.9134761, upper bound: 3611.9362720
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.9134761, upper bound: 3611.9362720
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -2182.4965820, 1870.5206299, -2057.4877930, 1791.9794922, -3974.4760742, 3928.0083008
1: -1757.9085693, 1830.2285156, -1651.0880127, 1757.4875488, -3515.3959961, 3481.3164062
2: -2598.9436035, 1985.5732422, -2419.5642090, 1902.2930908, -4501.2368164, 4405.1376953
3: -979.1383667, 2594.6718750, -936.3309937, 2457.2944336, -3436.4328613, 3531.0029297
4: -2859.6596680, 1935.0572510, -2664.9460449, 1852.7579346, -4712.4169922, 4600.0029297

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0215804, upper bound: 3612.5955662
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0215804, upper bound: 3612.5955611
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2435.3706055, 2091.4899902, -2147.6633301, 1866.9766846, -4302.3461914, 4239.1533203
1: -1961.2314453, 2048.3383789, -1723.0256348, 1830.5748291, -3791.8061523, 3771.3640137
2: -2895.4282227, 2220.5781250, -2524.4301758, 1981.3300781, -4876.7583008, 4745.0073242
3: -1094.4318848, 2902.0563965, -976.4069214, 2562.8596191, -3657.2915039, 3878.4633789
4: -3187.0346680, 2164.1335449, -2781.0791016, 1929.8286133, -5116.8632812, 4945.2114258

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0220227, upper bound: 3612.5955662
time: 1.15 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0220227, upper bound: 3612.5955662
time: 1.71 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2182.4545898, 1870.4897461, -2004.5377197, 1741.3880615, -3923.8422852, 3875.0273438
1: -1757.8740234, 1830.1986084, -1609.5473633, 1708.5133057, -3466.3872070, 3439.7456055
2: -2598.8906250, 1985.5410156, -2360.2414551, 1850.3981934, -4449.2890625, 4345.7822266
3: -979.1222534, 2594.6230469, -912.7960815, 2393.7290039, -3372.8513184, 3507.4189453
4: -2859.6018066, 1935.0251465, -2598.2106934, 1800.9992676, -4660.6000977, 4533.2358398

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0164047, upper bound: 3611.8425372
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0164047, upper bound: 3611.8425372
time: 1.08 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2435.3212891, 2091.4550781, -2059.3156738, 1787.4958496, -4222.8173828, 4150.7705078
1: -1961.1916504, 2048.3049316, -1653.2061768, 1753.3509521, -3714.5424805, 3701.5112305
2: -2895.3671875, 2220.5422363, -2423.8579102, 1898.7449951, -4794.1123047, 4644.3999023
3: -1094.4136963, 2902.0002441, -937.8716431, 2458.5400391, -3552.9533691, 3839.8718262
4: -3186.9680176, 2164.0974121, -2668.7690430, 1848.1281738, -5035.0961914, 4832.8657227

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0169532, upper bound: 3611.8425372
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0169532, upper bound: 3611.8425372
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2146.8740234, 1838.0312500, -2057.4877930, 1791.9794922, -3938.8535156, 3895.5187988
1: -1729.7517090, 1799.1739502, -1651.0880127, 1757.4875488, -3487.2387695, 3450.2617188
2: -2558.7514648, 1952.8004150, -2419.5642090, 1902.2930908, -4461.0444336, 4372.3647461
3: -964.2346191, 2553.4538574, -936.3309937, 2457.2944336, -3421.5288086, 3489.7844238
4: -2813.3908691, 1901.5114746, -2664.9460449, 1852.7579346, -4666.1484375, 4566.4575195

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.9232158, upper bound: 3612.2025809
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.9232158, upper bound: 3612.2025809
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2330.8544922, 1997.7227783, -2147.6633301, 1866.9766846, -4197.8310547, 4145.3862305
1: -1878.0645752, 1956.9641113, -1723.0256348, 1830.5748291, -3708.6394043, 3679.9897461
2: -2775.2331543, 2122.7390137, -2524.4301758, 1981.3300781, -4756.5625000, 4647.1689453
3: -1049.2624512, 2777.5441895, -976.4069214, 2562.8596191, -3612.1218262, 3753.9511719
4: -3053.0541992, 2067.6601562, -2781.0791016, 1929.8286133, -4982.8818359, 4848.7392578

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.9232158, upper bound: 3612.2046004
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.9232158, upper bound: 3612.2046004
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2146.8403320, 1838.0070801, -2004.5377197, 1741.3880615, -3888.2275391, 3842.5449219
1: -1729.7235107, 1799.1502686, -1609.5473633, 1708.5133057, -3438.2368164, 3408.6977539
2: -2558.7080078, 1952.7752686, -2360.2414551, 1850.3981934, -4409.1059570, 4313.0166016
3: -964.2222900, 2553.4138184, -912.7960815, 2393.7290039, -3357.9511719, 3466.2097168
4: -2813.3427734, 1901.4862061, -2598.2106934, 1800.9992676, -4614.3417969, 4499.6967773

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.9232158, upper bound: 3611.8425372
time: 0.86 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.9232158, upper bound: 3611.8425372
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2330.8161621, 1997.6954346, -2059.3156738, 1787.4958496, -4118.3120117, 4057.0112305
1: -1878.0329590, 1956.9373779, -1653.2061768, 1753.3509521, -3631.3837891, 3610.1435547
2: -2775.1840820, 2122.7102051, -2423.8579102, 1898.7449951, -4673.9291992, 4546.5678711
3: -1049.2481689, 2777.4997559, -937.8716431, 2458.5400391, -3507.7880859, 3715.3713379
4: -3053.0009766, 2067.6311035, -2668.7690430, 1848.1281738, -4901.1289062, 4736.3999023

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.9232158, upper bound: 3611.8425372
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.9232158, upper bound: 3611.8425372
time: 0.80 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 2.47 seconds
NS_A1_B2_B2_B1_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3611.5655836, upper bound: 3613.4711846
NS_A1_B2_B2_B1_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3611.5655836, upper bound: 3613.4734928
NS_A1_B2_B2_B1_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3612.4485931, upper bound: 3613.4711532
NS_A1_B2_B2_B1_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3612.4485931, upper bound: 3613.4737141
NS_A1_B2_B2_B1_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3611.5066817, upper bound: 3613.5041910
NS_A1_B2_B2_B1_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3611.5066817, upper bound: 3613.5145017
NS_A1_B2_B2_B1_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3611.7405594, upper bound: 3613.5118350
NS_A1_B2_B2_B1_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3611.7405594, upper bound: 3613.5145458
NS_A1_B2_B2_B1_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3611.5656478, upper bound: 3613.5121225
NS_A1_B2_B2_B1_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3611.5656478, upper bound: 3613.5143689
NS_A1_B2_B2_B1_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3612.4478959, upper bound: 3613.5122688
NS_A1_B2_B2_B1_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3612.4478959, upper bound: 3613.5146120
NS_A1_B2_B2_B2_A1_B1_A1_B1_B1, status: Status.VERIFIED, split count: 9, time: 2.47
Output dim: 0, lower bound: -3611.8518215, upper bound: 3612.0542885
NS_A1_B2_B2_B2_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3611.9472110, upper bound: 3614.0217299
NS_A1_B2_B2_B2_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3611.9456755, upper bound: 3613.5151710
NS_A1_B2_B2_B2_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3611.9471559, upper bound: 3614.0224362
NS_A1_B2_B2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3611.8926655, upper bound: 3614.0057492
NS_A1_B2_B2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3611.8926655, upper bound: 3614.0164681
NS_A1_B2_B2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3611.8927447, upper bound: 3614.0057432
NS_A1_B2_B2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3611.8927440, upper bound: 3614.0169967
NS_A1_B2_B2_B2_A1_B2_A1_B1_B1, status: Status.VERIFIED, split count: 9, time: 2.47
Output dim: 0, lower bound: -3611.8351743, upper bound: 3613.1177574
NS_A1_B2_B2_B2_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3611.8351743, upper bound: 3613.9232770
NS_A1_B2_B2_B2_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3611.8897573, upper bound: 3613.7739816
NS_A1_B2_B2_B2_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3611.8897573, upper bound: 3613.9232830
NS_A1_B2_B2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3611.9406430, upper bound: 3613.9244276
NS_A1_B2_B2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3611.9406422, upper bound: 3613.9244825
NS_A1_B2_B2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3611.9406422, upper bound: 3613.9244337
NS_A1_B2_B2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3611.9406422, upper bound: 3613.9244825
NS_A1_B2_B2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3612.5955662, upper bound: 3614.0215804
NS_A1_B2_B2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3612.5955662, upper bound: 3614.0217802
NS_A1_B2_B2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3612.5955662, upper bound: 3614.0220227
NS_A1_B2_B2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3612.5955662, upper bound: 3614.0224361
NS_A1_B2_B2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3611.8425372, upper bound: 3614.0164097
NS_A1_B2_B2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3611.8425372, upper bound: 3614.0164555
NS_A1_B2_B2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3611.8425372, upper bound: 3614.0169532
NS_A1_B2_B2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3611.8425372, upper bound: 3614.0169971
NS_A1_B2_B2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3612.2025809, upper bound: 3613.9232158
NS_A1_B2_B2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3612.2025809, upper bound: 3613.9232603
NS_A1_B2_B2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3612.2046004, upper bound: 3613.9232158
NS_A1_B2_B2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3612.2046004, upper bound: 3613.9232603
NS_A1_B2_B2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3611.8425372, upper bound: 3613.9239038
NS_A1_B2_B2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3611.8425372, upper bound: 3613.9238978
NS_A1_B2_B2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3611.8425372, upper bound: 3613.9238978
NS_A1_B2_B2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3611.8425372, upper bound: 3613.9239038
NS_A2_B1_A1_A2_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.7943225, upper bound: 3612.5661049
NS_A2_B1_A1_A2_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.7943225, upper bound: 3612.5661049
NS_A2_B1_A1_A2_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3614.0217199, upper bound: 3612.5662630
NS_A2_B1_A1_A2_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3614.0217199, upper bound: 3612.5662630
NS_A2_B1_A1_A2_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.5305352, upper bound: 3612.5714756
NS_A2_B1_A1_A2_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.5305352, upper bound: 3612.5790544
NS_A2_B1_A1_A2_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.5305352, upper bound: 3612.5243954
NS_A2_B1_A1_A2_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.5305352, upper bound: 3612.5414763
NS_A2_B1_A1_A2_B2_B2_A2_A1_A1, status: Status.VERIFIED, split count: 9, time: 2.47
Output dim: 0, lower bound: -3612.7783137, upper bound: 3612.5657832
NS_A2_B1_A1_A2_B2_B2_A2_A1_A2, status: Status.VERIFIED, split count: 9, time: 2.47
Output dim: 0, lower bound: -3612.7783137, upper bound: 3612.5661036
NS_A2_B1_A1_A2_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.6160654, upper bound: 3612.5659917
NS_A2_B1_A1_A2_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.6160654, upper bound: 3612.5662567
NS_A2_B1_A2_A1_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.4711872, upper bound: 3611.5064895
NS_A2_B1_A2_A1_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.4711872, upper bound: 3611.6767121
NS_A2_B1_A2_A1_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.4711726, upper bound: 3611.7403670
NS_A2_B1_A2_A1_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.4711726, upper bound: 3611.7599242
NS_A2_B1_A2_A1_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.4711793, upper bound: 3611.5655836
NS_A2_B1_A2_A1_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.4711793, upper bound: 3611.9150142
NS_A2_B1_A2_A1_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.4711542, upper bound: 3612.4485931
NS_A2_B1_A2_A1_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.4711542, upper bound: 3612.7354297
NS_A2_B1_A2_A1_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.5041921, upper bound: 3611.5066817
NS_A2_B1_A2_A1_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.5041921, upper bound: 3611.6769015
NS_A2_B1_A2_A1_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.5118350, upper bound: 3611.7405594
NS_A2_B1_A2_A1_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.5118350, upper bound: 3611.7601164
NS_A2_B1_A2_A1_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.5071075, upper bound: 3611.5657759
NS_A2_B1_A2_A1_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.5071075, upper bound: 3611.9152004
NS_A2_B1_A2_A1_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.5123020, upper bound: 3612.4396227
NS_A2_B1_A2_A1_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.5123020, upper bound: 3612.6272545
NS_A2_B1_A2_A2_B1_A1_B1_A1_A1, status: Status.VERIFIED, split count: 9, time: 2.47
Output dim: 0, lower bound: -3612.0542885, upper bound: 3611.8518215
NS_A2_B1_A2_A2_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3614.0217299, upper bound: 3611.9472110
NS_A2_B1_A2_A2_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.5151710, upper bound: 3611.9456755
NS_A2_B1_A2_A2_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3614.0224315, upper bound: 3611.9471559
NS_A2_B1_A2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3614.0057432, upper bound: 3611.8926655
NS_A2_B1_A2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3614.0057432, upper bound: 3611.8926662
NS_A2_B1_A2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3614.0057432, upper bound: 3611.8927447
NS_A2_B1_A2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3614.0057432, upper bound: 3611.8927440
NS_A2_B1_A2_A2_B1_A2_B1_A1_A1, status: Status.VERIFIED, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.1177550, upper bound: 3611.8351743
NS_A2_B1_A2_A2_B1_A2_B1_A1_A2, status: Status.VERIFIED, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.1177550, upper bound: 3611.8351743
NS_A2_B1_A2_A2_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.7739814, upper bound: 3611.8897573
NS_A2_B1_A2_A2_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.7739814, upper bound: 3611.8899841
NS_A2_B1_A2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.9134761, upper bound: 3611.9362720
NS_A2_B1_A2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.9134761, upper bound: 3611.9362720
NS_A2_B1_A2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.9134761, upper bound: 3611.9362720
NS_A2_B1_A2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.9134761, upper bound: 3611.9362720
NS_A2_B1_A2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3614.0215804, upper bound: 3612.5955662
NS_A2_B1_A2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3614.0215804, upper bound: 3612.5955611
NS_A2_B1_A2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3614.0220227, upper bound: 3612.5955662
NS_A2_B1_A2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3614.0220227, upper bound: 3612.5955662
NS_A2_B1_A2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3614.0164047, upper bound: 3611.8425372
NS_A2_B1_A2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3614.0164047, upper bound: 3611.8425372
NS_A2_B1_A2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3614.0169532, upper bound: 3611.8425372
NS_A2_B1_A2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3614.0169532, upper bound: 3611.8425372
NS_A2_B1_A2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.9232158, upper bound: 3612.2025809
NS_A2_B1_A2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.9232158, upper bound: 3612.2025809
NS_A2_B1_A2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.9232158, upper bound: 3612.2046004
NS_A2_B1_A2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.9232158, upper bound: 3612.2046004
NS_A2_B1_A2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.9232158, upper bound: 3611.8425372
NS_A2_B1_A2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.9232158, upper bound: 3611.8425372
NS_A2_B1_A2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.9232158, upper bound: 3611.8425372
NS_A2_B1_A2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.47
Output dim: 0, lower bound: -3613.9232158, upper bound: 3611.8425372

## BFS NS instance: NS_A1_B2_B2_B1_B1_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2205.1918945, 1931.8532715, -2473.8356934, 2141.2910156, -4346.4819336, 4405.6889648
1: -1770.2071533, 1895.9822998, -1992.5947266, 2097.5480957, -3867.7551270, 3888.5771484
2: -2596.3427734, 2052.0432129, -2944.5527344, 2274.0808105, -4870.4233398, 4996.5957031
3: -1006.9912109, 2641.8823242, -1116.6037598, 2955.7416992, -3962.7329102, 3758.4855957
4: -2858.1899414, 1999.9787598, -3239.3371582, 2218.9528809, -5077.1425781, 5239.3159180

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 43

## BFS NS instance: NS_A1_B2_B2_B1_B1_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2194.4704590, 1920.1577148, -2474.1064453, 2141.4938965, -4335.9643555, 4394.2641602
1: -1761.4803467, 1884.3338623, -1992.8167725, 2097.7448730, -3859.2241211, 3877.1499023
2: -2582.9672852, 2039.3208008, -2944.9006348, 2274.2927246, -4857.2597656, 4984.2216797
3: -1001.5255737, 2627.8344727, -1116.7073975, 2956.0583496, -3957.5837402, 3744.5417480
4: -2843.5258789, 1987.5223389, -3239.7170410, 2219.1652832, -5062.6899414, 5227.2392578

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 43

## BFS NS instance: NS_A1_B2_B2_B1_B1_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2123.7641602, 1849.3831787, -2473.8562012, 2141.3059082, -4265.0703125, 4323.2392578
1: -1705.4268799, 1815.0085449, -1992.6116943, 2097.5625000, -3802.9892578, 3807.6198730
2: -2501.5778809, 1964.3084717, -2944.5815430, 2274.0961914, -4775.6738281, 4908.8896484
3: -968.4545898, 2541.0310059, -1116.6107178, 2955.7663574, -3924.2209473, 3657.6416016
4: -2753.2182617, 1913.6060791, -3239.3681641, 2218.9685059, -4972.1865234, 5152.9741211

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 43

## BFS NS instance: NS_A1_B2_B2_B1_B1_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2115.1169434, 1840.5773926, -2474.1286621, 2141.5097656, -4256.6269531, 4314.7055664
1: -1698.2984619, 1806.5477295, -1992.8350830, 2097.7602539, -3796.0581055, 3799.3828125
2: -2490.5112305, 1956.1621094, -2944.9309082, 2274.3095703, -4764.8208008, 4901.0922852
3: -964.4971313, 2529.3359375, -1116.7150879, 2956.0847168, -3920.5817871, 3646.0507812
4: -2741.1928711, 1903.8385010, -3239.7497559, 2219.1823730, -4960.3750000, 5143.5883789

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 43

## BFS NS instance: NS_A1_B2_B2_B1_B2_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2182.2929688, 1916.6291504, -2490.7290039, 2150.8361816, -4333.1284180, 4407.3583984
1: -1751.9160156, 1880.9443359, -2007.0411377, 2106.6022949, -3858.5183105, 3887.9853516
2: -2570.1550293, 2036.5375977, -2965.7485352, 2283.8083496, -4853.9633789, 5002.2856445
3: -997.7923584, 2615.7841797, -1123.7739258, 2976.4892578, -3974.2812500, 3739.5581055
4: -2829.2761230, 1984.1448975, -3261.6606445, 2228.3488770, -5057.6245117, 5245.8046875

Time for backsubstitution: 0.80 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.55 + 418.09 = 420.64 seconds
