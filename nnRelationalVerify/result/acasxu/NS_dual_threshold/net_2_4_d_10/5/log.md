## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 2331.289411072758


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-552.1849365, 2165.8994141, -552.1849365, 2165.8994141, -2718.0837402, 2718.0837402)
1: (-380.9231873, 1079.1445312, -380.9231873, 1079.1445312, -1460.0673828, 1460.0673828)
2: (-214.0630951, 918.4811401, -214.0630951, 918.4811401, -1132.5441895, 1132.5441895)
3: (-269.2298889, 1565.5185547, -269.2298889, 1565.5185547, -1834.7484131, 1834.7484131)
4: (-368.3156738, 1214.9196777, -368.3156738, 1214.9196777, -1583.2353516, 1583.2353516)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.85 + 2.07 = 3.92 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -2331.3127242, upper bound: 2331.3127242

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3109690, upper bound: 2331.3116670
time: 1.05 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3109665, upper bound: 2331.3109665
time: 1.01 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.22 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.22
Output dim: 0, lower bound: -2331.3109690, upper bound: 2331.3116670
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.22
Output dim: 0, lower bound: -2331.3109665, upper bound: 2331.3109665

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -522.8555298, 2046.0078125, -535.2879639, 2096.8371582, -2619.6926270, 2581.2958984
1: -360.1157837, 1021.5060425, -368.9321594, 1045.9050293, -1406.0207520, 1390.4382324
2: -202.4834442, 869.3995361, -207.3911896, 890.1712036, -1092.6546631, 1076.7905273
3: -254.5275574, 1481.8624268, -260.7515564, 1517.2761230, -1771.8037109, 1742.6138916
4: -348.4825745, 1149.6829834, -356.8826904, 1177.2907715, -1525.7733154, 1506.5656738

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3109665, upper bound: 2331.3109665
time: 0.77 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3109665, upper bound: 2331.3109665
time: 0.75 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -614.7044067, 2429.7268066, -536.0252075, 2102.3757324, -2717.0800781, 2965.7519531
1: -426.6650391, 1203.3717041, -369.7273560, 1046.8341064, -1473.4991455, 1573.0989990
2: -239.5143738, 1024.7359619, -207.7450409, 891.0041504, -1130.5183105, 1232.4809570
3: -301.9007568, 1745.4802246, -261.2552490, 1518.5915527, -1820.4923096, 2006.7354736
4: -411.4239807, 1356.5231934, -357.3338623, 1178.4379883, -1589.8619385, 1713.8570557

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3109665, upper bound: 2331.3109665
time: 0.86 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3109665, upper bound: 2331.3109665
time: 0.75 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.49 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.49
Output dim: 0, lower bound: -2331.3109665, upper bound: 2331.3109665
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.49
Output dim: 0, lower bound: -2331.3109665, upper bound: 2331.3109665
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.49
Output dim: 0, lower bound: -2331.3109665, upper bound: 2331.3109665
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.49
Output dim: 0, lower bound: -2331.3109665, upper bound: 2331.3109665

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -522.8555298, 2046.0078125, -522.8555298, 2046.0078125, -2568.8632812, 2568.8632812
1: -360.1157837, 1021.5060425, -360.1157837, 1021.5060425, -1381.6218262, 1381.6218262
2: -202.4834442, 869.3995361, -202.4834442, 869.3995361, -1071.8829346, 1071.8829346
3: -254.5275574, 1481.8624268, -254.5275574, 1481.8624268, -1736.3898926, 1736.3898926
4: -348.4825745, 1149.6829834, -348.4825745, 1149.6829834, -1498.1655273, 1498.1655273

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3077338, upper bound: 2331.3089224
time: 0.90 seconds

## Relational analysis of NS_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3067266, upper bound: 2331.3078198
time: 0.80 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -522.8555298, 2046.0078125, -614.7044067, 2429.7268066, -2952.5822754, 2660.7121582
1: -360.1157837, 1021.5060425, -426.6650391, 1203.3717041, -1563.4874268, 1448.1711426
2: -202.4834442, 869.3995361, -239.5143738, 1024.7359619, -1227.2193604, 1108.9139404
3: -254.5275574, 1481.8624268, -301.9007568, 1745.4802246, -2000.0078125, 1783.7630615
4: -348.4825745, 1149.6829834, -411.4239807, 1356.5231934, -1705.0057373, 1561.1069336

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3085299, upper bound: 2331.3099385
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3083082, upper bound: 2331.3098900
time: 1.03 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -614.7044067, 2429.7268066, -522.8555298, 2046.0078125, -2660.7121582, 2952.5822754
1: -426.6650391, 1203.3717041, -360.1157837, 1021.5060425, -1448.1711426, 1563.4874268
2: -239.5143738, 1024.7359619, -202.4834442, 869.3995361, -1108.9139404, 1227.2193604
3: -301.9007568, 1745.4802246, -254.5275574, 1481.8624268, -1783.7630615, 2000.0078125
4: -411.4239807, 1356.5231934, -348.4825745, 1149.6829834, -1561.1069336, 1705.0057373

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3092136, upper bound: 2331.3085299
time: 0.81 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3083082, upper bound: 2331.3083082
time: 0.77 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -614.7044067, 2429.7268066, -614.7044067, 2429.7268066, -3044.4309082, 3044.4309082
1: -426.6650391, 1203.3717041, -426.6650391, 1203.3717041, -1630.0367432, 1630.0367432
2: -239.5143738, 1024.7359619, -239.5143738, 1024.7359619, -1264.2503662, 1264.2503662
3: -301.9007568, 1745.4802246, -301.9007568, 1745.4802246, -2047.3809814, 2047.3809814
4: -411.4239807, 1356.5231934, -411.4239807, 1356.5231934, -1767.9471436, 1767.9471436

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3092136, upper bound: 2331.3085299
time: 0.76 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3083082, upper bound: 2331.3083082
time: 0.74 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.39 seconds
NS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.39
Output dim: 0, lower bound: -2331.3077338, upper bound: 2331.3089224
NS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.39
Output dim: 0, lower bound: -2331.3067266, upper bound: 2331.3078198
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.39
Output dim: 0, lower bound: -2331.3085299, upper bound: 2331.3099385
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.39
Output dim: 0, lower bound: -2331.3083082, upper bound: 2331.3098900
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.39
Output dim: 0, lower bound: -2331.3092136, upper bound: 2331.3085299
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.39
Output dim: 0, lower bound: -2331.3083082, upper bound: 2331.3083082
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.39
Output dim: 0, lower bound: -2331.3092136, upper bound: 2331.3085299
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.39
Output dim: 0, lower bound: -2331.3083082, upper bound: 2331.3083082

## BFS NS instance: NS_A1_B1_B1

### Backsubstitution after applying NS history:
0: -522.8555298, 2046.0078125, -510.6761780, 1997.0258789, -2519.8813477, 2556.6835938
1: -360.1157837, 1021.5060425, -351.6681519, 997.4882202, -1357.6040039, 1373.1741943
2: -202.4834442, 869.3995361, -197.7324219, 848.9814453, -1051.4647217, 1067.1318359
3: -254.5275574, 1481.8624268, -248.5334167, 1447.1052246, -1701.6328125, 1730.3957520
4: -348.4825745, 1149.6829834, -340.2532654, 1122.6528320, -1471.1353760, 1489.9362793

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3078198, upper bound: 2331.3078198
time: 1.16 seconds

## Relational analysis of NS_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3078198, upper bound: 2331.3078198
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_B2

### Backsubstitution after applying NS history:
0: -510.1271973, 1994.5477295, -571.5869751, 2223.4489746, -2733.5761719, 2566.1347656
1: -351.2234192, 996.5530396, -392.4953308, 1112.9425049, -1464.1657715, 1389.0483398
2: -197.5534058, 848.1322632, -220.9544373, 946.0488892, -1143.6022949, 1069.0866699
3: -248.3003998, 1445.6574707, -278.2347107, 1616.8702393, -1865.1701660, 1723.8922119
4: -340.0464783, 1121.4660645, -380.8812256, 1253.0339355, -1593.0803223, 1502.3472900

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3078198, upper bound: 2331.3078198
time: 0.72 seconds

## Relational analysis of NS_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3078198, upper bound: 2331.3078198
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -467.3208313, 1830.5625000, -579.0029907, 2292.4785156, -2759.7988281, 2409.5642090
1: -322.1680908, 912.9301147, -402.3046265, 1133.3649902, -1455.5330811, 1315.2346191
2: -180.9830627, 777.0300903, -225.7222443, 965.0961914, -1146.0792236, 1002.7523193
3: -227.5158081, 1324.6269531, -284.6502075, 1644.0594482, -1871.5751953, 1609.2770996
4: -311.6451416, 1027.4747314, -387.6279297, 1277.6339111, -1589.2790527, 1415.1024170

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3083082, upper bound: 2331.3098694
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3083082, upper bound: 2331.3098694
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -595.3701782, 2346.1115723, -597.8720093, 2364.3713379, -2959.7414551, 2943.9831543
1: -410.8760376, 1158.4462891, -415.1936951, 1171.0528564, -1581.9289551, 1573.6400146
2: -230.8953857, 986.6727905, -233.1094513, 997.2326660, -1228.1276855, 1219.7821045
3: -289.4539490, 1681.1589355, -293.8873291, 1698.5825195, -1988.0364990, 1975.0462646
4: -396.5617065, 1307.1392822, -400.4698181, 1320.0310059, -1716.5927734, 1707.6091309

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3083082, upper bound: 2331.3098900
time: 1.04 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3083082, upper bound: 2331.3098900
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -579.0029907, 2292.4785156, -467.3208313, 1830.5625000, -2409.5644531, 2759.7990723
1: -402.3046265, 1133.3649902, -322.1680908, 912.9301147, -1315.2346191, 1455.5330811
2: -225.7222443, 965.0961914, -180.9830627, 777.0300903, -1002.7523193, 1146.0792236
3: -284.6502075, 1644.0594482, -227.5158081, 1324.6269531, -1609.2770996, 1871.5751953
4: -387.6279297, 1277.6339111, -311.6451416, 1027.4747314, -1415.1024170, 1589.2790527

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3098694, upper bound: 2331.3083082
time: 0.72 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3098694, upper bound: 2331.3083082
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -597.8720093, 2364.3713379, -595.3701782, 2346.1115723, -2943.9831543, 2959.7414551
1: -415.1936951, 1171.0528564, -410.8760376, 1158.4462891, -1573.6400146, 1581.9289551
2: -233.1094513, 997.2326660, -230.8953857, 986.6727905, -1219.7821045, 1228.1276855
3: -293.8873291, 1698.5825195, -289.4539490, 1681.1589355, -1975.0462646, 1988.0364990
4: -400.4698181, 1320.0310059, -396.5617065, 1307.1392822, -1707.6091309, 1716.5927734

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3098900, upper bound: 2331.3083082
time: 1.08 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3098900, upper bound: 2331.3083082
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: -579.0029907, 2292.4785156, -550.7278442, 2183.4577637, -2762.4602051, 2843.2062988
1: -402.3046265, 1133.3649902, -382.9767456, 1078.0111084, -1480.3156738, 1516.3417969
2: -225.7222443, 965.0961914, -214.8183594, 918.1005249, -1143.8222656, 1179.9145508
3: -284.6502075, 1644.0594482, -270.9054260, 1563.8934326, -1848.5437012, 1914.9648438
4: -387.6279297, 1277.6339111, -368.7893677, 1215.5390625, -1603.1669922, 1646.4233398

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3083082, upper bound: 2331.3083082
time: 0.87 seconds

## Relational analysis of NS_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3083082, upper bound: 2331.3083082
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: -597.8720093, 2364.3713379, -680.3256226, 2695.0239258, -3292.8957520, 3044.6970215
1: -415.1936951, 1171.0528564, -471.7484131, 1329.2425537, -1744.4362793, 1642.8011475
2: -233.1094513, 997.2326660, -264.7478638, 1132.2117920, -1365.3210449, 1261.9803467
3: -293.8873291, 1698.5825195, -333.3937683, 1928.4042969, -2222.2915039, 2031.9763184
4: -400.4698181, 1320.0310059, -454.6759949, 1499.5177002, -1899.9875488, 1774.7070312

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3083082, upper bound: 2331.3083082
time: 0.80 seconds

## Relational analysis of NS_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3083082, upper bound: 2331.3083082
time: 0.71 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 5.68 seconds
NS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 5.68
Output dim: 0, lower bound: -2331.3078198, upper bound: 2331.3078198
NS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 5.68
Output dim: 0, lower bound: -2331.3078198, upper bound: 2331.3078198
NS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 5.68
Output dim: 0, lower bound: -2331.3078198, upper bound: 2331.3078198
NS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 5.68
Output dim: 0, lower bound: -2331.3078198, upper bound: 2331.3078198
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.68
Output dim: 0, lower bound: -2331.3083082, upper bound: 2331.3098694
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.68
Output dim: 0, lower bound: -2331.3083082, upper bound: 2331.3098694
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.68
Output dim: 0, lower bound: -2331.3083082, upper bound: 2331.3098900
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.68
Output dim: 0, lower bound: -2331.3083082, upper bound: 2331.3098900
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 5.68
Output dim: 0, lower bound: -2331.3098694, upper bound: 2331.3083082
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 5.68
Output dim: 0, lower bound: -2331.3098694, upper bound: 2331.3083082
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 5.68
Output dim: 0, lower bound: -2331.3098900, upper bound: 2331.3083082
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 5.68
Output dim: 0, lower bound: -2331.3098900, upper bound: 2331.3083082
NS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 5.68
Output dim: 0, lower bound: -2331.3083082, upper bound: 2331.3083082
NS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 5.68
Output dim: 0, lower bound: -2331.3083082, upper bound: 2331.3083082
NS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 5.68
Output dim: 0, lower bound: -2331.3083082, upper bound: 2331.3083082
NS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 5.68
Output dim: 0, lower bound: -2331.3083082, upper bound: 2331.3083082

## BFS NS instance: NS_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -510.6761780, 1997.0258789, -510.6761780, 1997.0258789, -2507.7014160, 2507.7016602
1: -351.6681519, 997.4882202, -351.6681519, 997.4882202, -1349.1563721, 1349.1563721
2: -197.7324219, 848.9814453, -197.7324219, 848.9814453, -1046.7136230, 1046.7136230
3: -248.5334167, 1447.1052246, -248.5334167, 1447.1052246, -1695.6386719, 1695.6386719
4: -340.2532654, 1122.6528320, -340.2532654, 1122.6528320, -1462.9061279, 1462.9061279

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A1_A1

### Relational analysis result of NS_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3062065, upper bound: 2331.3060090
time: 0.82 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2

### Relational analysis result of NS_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3082077, upper bound: 2331.3089198
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -571.5869751, 2223.4489746, -510.6761780, 1997.0258789, -2568.6127930, 2734.1245117
1: -392.4953308, 1112.9425049, -351.6681519, 997.4882202, -1389.9835205, 1464.6105957
2: -220.9544373, 946.0488892, -197.7324219, 848.9814453, -1069.9356689, 1143.7810059
3: -278.2347107, 1616.8702393, -248.5334167, 1447.1052246, -1725.3399658, 1865.4034424
4: -380.8812256, 1253.0339355, -340.2532654, 1122.6528320, -1503.5340576, 1593.2872314

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_B1_A2_B1

### Relational analysis result of NS_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.2991301, upper bound: 2331.2981543
time: 0.75 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2

### Relational analysis result of NS_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3084847, upper bound: 2331.3089224
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -510.6761780, 1997.0258789, -571.5869751, 2223.4489746, -2734.1245117, 2568.6127930
1: -351.6681519, 997.4882202, -392.4953308, 1112.9425049, -1464.6105957, 1389.9835205
2: -197.7324219, 848.9814453, -220.9544373, 946.0488892, -1143.7810059, 1069.9356689
3: -248.5334167, 1447.1052246, -278.2347107, 1616.8702393, -1865.4035645, 1725.3399658
4: -340.2532654, 1122.6528320, -380.8812256, 1253.0339355, -1593.2872314, 1503.5340576

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_B2_A1_B1

### Relational analysis result of NS_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3062699, upper bound: 2331.3045337
time: 0.73 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2

### Relational analysis result of NS_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3078198, upper bound: 2331.3078198
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -571.5869751, 2223.4489746, -571.5869751, 2223.4489746, -2794.9074707, 2794.9074707
1: -392.4953308, 1112.9425049, -392.4953308, 1112.9425049, -1505.4378662, 1505.4378662
2: -220.9544373, 946.0488892, -220.9544373, 946.0488892, -1167.0031738, 1167.0031738
3: -278.2347107, 1616.8702393, -278.2347107, 1616.8702393, -1895.1049805, 1895.1049805
4: -380.8812256, 1253.0339355, -380.8812256, 1253.0339355, -1633.9151611, 1633.9151611

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_B2_A2_A1

### Relational analysis result of NS_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3076143, upper bound: 2331.3055231
time: 0.91 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2

### Relational analysis result of NS_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3076143, upper bound: 2331.3076143
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -467.3208313, 1830.5625000, -550.7278442, 2183.4577637, -2650.7785645, 2381.2897949
1: -322.1680908, 912.9301147, -382.9767456, 1078.0111084, -1400.1791992, 1295.9068604
2: -180.9830627, 777.0300903, -214.8183594, 918.1005249, -1099.0833740, 991.8484497
3: -227.5158081, 1324.6269531, -270.9054260, 1563.8934326, -1791.4091797, 1595.5323486
4: -311.6451416, 1027.4747314, -368.7893677, 1215.5390625, -1527.1842041, 1396.2639160

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B1_B1

### Relational analysis result of NS_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3051400, upper bound: 2331.3058197
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2

### Relational analysis result of NS_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3043172, upper bound: 2331.3053624
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -467.3208313, 1830.5625000, -680.3256226, 2695.0239258, -3162.3447266, 2510.8876953
1: -322.1680908, 912.9301147, -471.7484131, 1329.2425537, -1651.4105225, 1384.6783447
2: -180.9830627, 777.0300903, -264.7478638, 1132.2117920, -1313.1947021, 1041.7779541
3: -227.5158081, 1324.6269531, -333.3937683, 1928.4042969, -2155.9199219, 1658.0207520
4: -311.6451416, 1027.4747314, -454.6759949, 1499.5177002, -1811.1628418, 1482.1506348

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3051400, upper bound: 2331.3058197
time: 1.05 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3043172, upper bound: 2331.3053624
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -595.3701782, 2346.1115723, -550.7278442, 2183.4577637, -2778.8278809, 2896.8393555
1: -410.8760376, 1158.4462891, -382.9767456, 1078.0111084, -1488.8872070, 1541.4230957
2: -230.8953857, 986.6727905, -214.8183594, 918.1005249, -1148.9957275, 1201.4912109
3: -289.4539490, 1681.1589355, -270.9054260, 1563.8934326, -1853.3474121, 1952.0643311
4: -396.5617065, 1307.1392822, -368.7893677, 1215.5390625, -1612.1007080, 1675.9287109

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3070205, upper bound: 2331.3078180
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3057703, upper bound: 2331.3074761
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -595.3701782, 2346.1115723, -680.3256226, 2695.0239258, -3290.3940430, 3026.4370117
1: -410.8760376, 1158.4462891, -471.7484131, 1329.2425537, -1740.1185303, 1630.1944580
2: -230.8953857, 986.6727905, -264.7478638, 1132.2117920, -1363.1068115, 1251.4205322
3: -289.4539490, 1681.1589355, -333.3937683, 1928.4042969, -2217.8581543, 2014.5527344
4: -396.5617065, 1307.1392822, -454.6759949, 1499.5177002, -1896.0793457, 1761.8153076

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B2_B1

### Relational analysis result of NS_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3069711, upper bound: 2331.3089198
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2

### Relational analysis result of NS_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3057703, upper bound: 2331.3078186
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -550.7278442, 2183.4577637, -467.3208313, 1830.5625000, -2381.2897949, 2650.7785645
1: -382.9767456, 1078.0111084, -322.1680908, 912.9301147, -1295.9068604, 1400.1791992
2: -214.8183594, 918.1005249, -180.9830627, 777.0300903, -991.8484497, 1099.0834961
3: -270.9054260, 1563.8934326, -227.5158081, 1324.6269531, -1595.5323486, 1791.4091797
4: -368.7893677, 1215.5390625, -311.6451416, 1027.4747314, -1396.2639160, 1527.1842041

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_B1_A1_A1

### Relational analysis result of NS_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3058197, upper bound: 2331.3051400
time: 0.76 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2

### Relational analysis result of NS_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3053624, upper bound: 2331.3043172
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -680.3256226, 2695.0239258, -467.3208313, 1830.5625000, -2510.8874512, 3162.3447266
1: -471.7484131, 1329.2425537, -322.1680908, 912.9301147, -1384.6783447, 1651.4105225
2: -264.7478638, 1132.2117920, -180.9830627, 777.0300903, -1041.7779541, 1313.1948242
3: -333.3937683, 1928.4042969, -227.5158081, 1324.6269531, -1658.0207520, 2155.9199219
4: -454.6759949, 1499.5177002, -311.6451416, 1027.4747314, -1482.1506348, 1811.1628418

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_B1_A2_A1

### Relational analysis result of NS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3058197, upper bound: 2331.3051400
time: 0.72 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2

### Relational analysis result of NS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3053624, upper bound: 2331.3048864
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -550.7278442, 2183.4577637, -595.3701782, 2346.1115723, -2896.8393555, 2778.8278809
1: -382.9767456, 1078.0111084, -410.8760376, 1158.4462891, -1541.4230957, 1488.8872070
2: -214.8183594, 918.1005249, -230.8953857, 986.6727905, -1201.4912109, 1148.9956055
3: -270.9054260, 1563.8934326, -289.4539490, 1681.1589355, -1952.0643311, 1853.3474121
4: -368.7893677, 1215.5390625, -396.5617065, 1307.1392822, -1675.9287109, 1612.1007080

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3078180, upper bound: 2331.3070205
time: 1.07 seconds

## Relational analysis of NS_A2_B1_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3053624, upper bound: 2331.3057703
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -680.3256226, 2695.0239258, -595.3701782, 2346.1115723, -3026.4370117, 3290.3940430
1: -471.7484131, 1329.2425537, -410.8760376, 1158.4462891, -1630.1944580, 1740.1185303
2: -264.7478638, 1132.2117920, -230.8953857, 986.6727905, -1251.4205322, 1363.1068115
3: -333.3937683, 1928.4042969, -289.4539490, 1681.1589355, -2014.5527344, 2217.8581543
4: -454.6759949, 1499.5177002, -396.5617065, 1307.1392822, -1761.8153076, 1896.0793457

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_B2_A2_A1

### Relational analysis result of NS_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3058197, upper bound: 2331.3065110
time: 0.80 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2

### Relational analysis result of NS_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3053624, upper bound: 2331.3064821
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -550.7278442, 2183.4577637, -550.7278442, 2183.4577637, -2734.1855469, 2734.1855469
1: -382.9767456, 1078.0111084, -382.9767456, 1078.0111084, -1460.9877930, 1460.9877930
2: -214.8183594, 918.1005249, -214.8183594, 918.1005249, -1132.9188232, 1132.9188232
3: -270.9054260, 1563.8934326, -270.9054260, 1563.8934326, -1834.7988281, 1834.7988281
4: -368.7893677, 1215.5390625, -368.7893677, 1215.5390625, -1584.3283691, 1584.3283691

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_B1_A1_A1

### Relational analysis result of NS_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.2970300, upper bound: 2331.2971203
time: 0.71 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_B1_A1_A1

### Relational analysis result of NS_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3065609, upper bound: 2331.3061086
time: 0.92 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2

### Relational analysis result of NS_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3061335, upper bound: 2331.3059101
time: 0.98 seconds

## BFS NS instance: NS_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -680.3256226, 2695.0239258, -550.7278442, 2183.4577637, -2863.7834473, 3245.7517090
1: -471.7484131, 1329.2425537, -382.9767456, 1078.0111084, -1549.7595215, 1712.2192383
2: -264.7478638, 1132.2117920, -214.8183594, 918.1005249, -1182.8483887, 1347.0301514
3: -333.3937683, 1928.4042969, -270.9054260, 1563.8934326, -1897.2872314, 2199.3098145
4: -454.6759949, 1499.5177002, -368.7893677, 1215.5390625, -1670.2150879, 1868.3071289

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_B1_A2_A1

### Relational analysis result of NS_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3065609, upper bound: 2331.3061086
time: 1.01 seconds

## Relational analysis of NS_A2_B2_B1_A2_A2

### Relational analysis result of NS_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3061335, upper bound: 2331.3059420
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -550.7278442, 2183.4577637, -680.3256226, 2695.0239258, -3245.7517090, 2863.7834473
1: -382.9767456, 1078.0111084, -471.7484131, 1329.2425537, -1712.2192383, 1549.7595215
2: -214.8183594, 918.1005249, -264.7478638, 1132.2117920, -1347.0301514, 1182.8481445
3: -270.9054260, 1563.8934326, -333.3937683, 1928.4042969, -2199.3098145, 1897.2872314
4: -368.7893677, 1215.5390625, -454.6759949, 1499.5177002, -1868.3071289, 1670.2150879

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_B2_A1_B1

### Relational analysis result of NS_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3073680, upper bound: 2331.3069718
time: 0.75 seconds

## Relational analysis of NS_A2_B2_B2_A1_B2

### Relational analysis result of NS_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3061335, upper bound: 2331.3060395
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -680.3256226, 2695.0239258, -680.3256226, 2695.0239258, -3375.3496094, 3375.3496094
1: -471.7484131, 1329.2425537, -471.7484131, 1329.2425537, -1800.9907227, 1800.9907227
2: -264.7478638, 1132.2117920, -264.7478638, 1132.2117920, -1396.9594727, 1396.9594727
3: -333.3937683, 1928.4042969, -333.3937683, 1928.4042969, -2261.7978516, 2261.7978516
4: -454.6759949, 1499.5177002, -454.6759949, 1499.5177002, -1954.1937256, 1954.1937256

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_B2_A2_B1

### Relational analysis result of NS_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3073680, upper bound: 2331.3077673
time: 1.12 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2

### Relational analysis result of NS_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3061335, upper bound: 2331.3064821
time: 0.89 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.98 seconds
NS_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -2331.3062065, upper bound: 2331.3060090
NS_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -2331.3082077, upper bound: 2331.3089198
NS_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -2331.2991301, upper bound: 2331.2981543
NS_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -2331.3084847, upper bound: 2331.3089224
NS_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -2331.3062699, upper bound: 2331.3045337
NS_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -2331.3078198, upper bound: 2331.3078198
NS_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -2331.3076143, upper bound: 2331.3055231
NS_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -2331.3076143, upper bound: 2331.3076143
NS_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -2331.3051400, upper bound: 2331.3058197
NS_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -2331.3043172, upper bound: 2331.3053624
NS_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -2331.3051400, upper bound: 2331.3058197
NS_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -2331.3043172, upper bound: 2331.3053624
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -2331.3070205, upper bound: 2331.3078180
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -2331.3057703, upper bound: 2331.3074761
NS_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -2331.3069711, upper bound: 2331.3089198
NS_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -2331.3057703, upper bound: 2331.3078186
NS_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -2331.3058197, upper bound: 2331.3051400
NS_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -2331.3053624, upper bound: 2331.3043172
NS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -2331.3058197, upper bound: 2331.3051400
NS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -2331.3053624, upper bound: 2331.3048864
NS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -2331.3078180, upper bound: 2331.3070205
NS_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -2331.3053624, upper bound: 2331.3057703
NS_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -2331.3058197, upper bound: 2331.3065110
NS_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -2331.3053624, upper bound: 2331.3064821
NS_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -2331.3065609, upper bound: 2331.3061086
NS_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -2331.3061335, upper bound: 2331.3059101
NS_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -2331.3065609, upper bound: 2331.3061086
NS_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -2331.3061335, upper bound: 2331.3059420
NS_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -2331.3073680, upper bound: 2331.3069718
NS_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -2331.3061335, upper bound: 2331.3060395
NS_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -2331.3073680, upper bound: 2331.3077673
NS_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -2331.3061335, upper bound: 2331.3064821

## BFS NS instance: NS_A1_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -456.4900513, 1786.6064453, -478.2996216, 1871.7889404, -2328.2790527, 2264.9060059
1: -314.6244202, 891.6278687, -329.5334473, 934.1475220, -1248.7717285, 1221.1613770
2: -176.7469788, 758.8996582, -185.2100983, 795.1245728, -971.8715820, 944.1096802
3: -222.1472168, 1293.7668457, -232.8345490, 1355.2792969, -1577.4265137, 1526.6014404
4: -304.3231812, 1003.4425049, -318.7889099, 1051.3438721, -1355.6669922, 1322.2313232

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A1_A1_B1

### Relational analysis result of NS_A1_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3111527, upper bound: 2331.3111527
time: 0.86 seconds

## Relational analysis of NS_A1_B1_B1_A1_A1_B2

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3111527, upper bound: 2331.3111527
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -584.4433594, 2302.2429199, -484.9971924, 1900.4077148, -2484.8510742, 2787.2402344
1: -403.3268127, 1136.8691406, -334.8075867, 948.1767578, -1351.5032959, 1471.6767578
2: -226.6615906, 968.3230591, -188.2700348, 807.5300293, -1034.1916504, 1156.5931396
3: -284.1044922, 1649.8958740, -236.6371002, 1375.8227539, -1659.9271240, 1886.5329590
4: -389.2089233, 1282.8598633, -323.7267761, 1068.2731934, -1457.4819336, 1606.5866699

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_B1_A1_A2_B1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3111910, upper bound: 2331.3110044
time: 1.00 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3110501, upper bound: 2331.3110501
time: 1.33 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -556.3983154, 2162.8825684, -479.0498047, 1870.4090576, -2426.8073730, 2641.9323730
1: -382.0004578, 1083.0039062, -329.7516174, 935.4185791, -1317.4189453, 1412.7554932
2: -215.0446320, 920.5657959, -185.3780060, 796.2625122, -1011.3071289, 1105.9438477
3: -270.7780457, 1573.5660400, -232.8432465, 1357.0957031, -1627.8737793, 1806.4089355
4: -370.6346741, 1219.3776855, -318.9134216, 1052.7302246, -1423.3648682, 1538.2911377

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.2989240, upper bound: 2331.2981543
time: 0.81 seconds

## Relational analysis of NS_A1_B1_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.2987620, upper bound: 2331.2979605
time: 0.74 seconds

## Relational analysis of NS_A1_B1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.2966682, upper bound: 2331.2979984
time: 1.03 seconds

## Relational analysis of NS_A1_B1_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 22

## BFS NS instance: NS_A1_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -558.1690063, 2169.8125000, -522.1117554, 2039.0766602, -2597.2456055, 2691.9240723
1: -383.2145386, 1086.8524170, -359.8381348, 1019.3985596, -1402.6130371, 1446.6904297
2: -215.7645264, 923.9137573, -202.5764771, 867.8295288, -1083.5939941, 1126.4898682
3: -271.5882568, 1578.8847656, -254.5616608, 1479.2614746, -1750.8497314, 1833.4464111
4: -371.9129639, 1223.5178223, -348.3152771, 1148.5153809, -1520.4281006, 1571.8331299

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3061371, upper bound: 2331.3060057
time: 1.19 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3082077, upper bound: 2331.3089198
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -505.1476135, 1974.8743896, -560.0466309, 2177.1188965, -2682.2656250, 2534.9208984
1: -347.7561951, 986.6710815, -384.4001160, 1090.2777100, -1438.0339355, 1371.0711670
2: -195.5351105, 839.7895508, -216.4344482, 926.7503662, -1122.2852783, 1056.2238770
3: -245.7579956, 1431.3792725, -272.5014648, 1584.0445557, -1829.8023682, 1703.8807373
4: -336.4895630, 1110.4062500, -373.0861206, 1227.4240723, -1563.9135742, 1483.4921875

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_B2_A1_B1_B1

### Relational analysis result of NS_A1_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3067910, upper bound: 2331.3034408
time: 0.87 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3068090, upper bound: 2331.3037879
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -501.9677429, 1960.8774414, -584.7406616, 2270.1000977, -2771.8745117, 2545.6179199
1: -345.3750916, 980.3453369, -400.5130005, 1137.9537354, -1483.3288574, 1380.8579102
2: -194.2959747, 834.3765259, -225.7321930, 967.1850586, -1161.4810791, 1060.1086426
3: -244.1515503, 1422.1544189, -283.8978882, 1652.8796387, -1897.0310059, 1706.0522461
4: -334.3714905, 1103.2319336, -389.2696533, 1280.7592773, -1615.1307373, 1492.5014648

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3089198, upper bound: 2331.3077128
time: 1.25 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3089198, upper bound: 2331.3081594
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -568.5019531, 2211.4204102, -570.9525757, 2220.9765625, -2789.3574219, 2782.2456055
1: -390.3895874, 1107.0008545, -392.0623169, 1111.7116699, -1502.1013184, 1499.0629883
2: -219.7639313, 940.9892578, -220.7092133, 945.0016479, -1164.7656250, 1161.6984863
3: -276.7389832, 1608.2596436, -277.9265747, 1615.0885010, -1891.8275146, 1886.1862793
4: -378.8573914, 1246.3210449, -380.4620972, 1251.6462402, -1630.5036621, 1626.7832031

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3055230, upper bound: 2331.3055230
time: 0.99 seconds

## Relational analysis of NS_A1_B1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3055231, upper bound: 2331.3055231
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -580.3777466, 2257.6291504, -568.5504761, 2211.4741211, -2791.6958008, 2826.0412598
1: -398.8233032, 1130.7194824, -390.3787537, 1106.9423828, -1505.7655029, 1521.0982666
2: -224.6190796, 961.3665161, -219.7707825, 940.9319458, -1165.5510254, 1181.1370850
3: -282.7142944, 1642.8055420, -276.7415466, 1608.1851807, -1890.8994141, 1919.5471191
4: -387.0067139, 1273.2871094, -378.8380737, 1246.2724609, -1633.2790527, 1652.1252441

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3040348, upper bound: 2331.3074064
time: 0.74 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3038786, upper bound: 2331.3030736
time: 1.06 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -467.3208313, 1830.5625000, -540.3713379, 2141.8784180, -2609.1992188, 2370.9338379
1: -322.1680908, 912.9301147, -375.7645264, 1057.5384521, -1379.7065430, 1288.6945801
2: -180.9830627, 777.0300903, -210.7732391, 900.6857300, -1081.6687012, 987.8032837
3: -227.5158081, 1324.6269531, -265.7885742, 1534.2711182, -1761.7868652, 1590.4155273
4: -311.6451416, 1027.4747314, -361.7720032, 1192.5009766, -1504.1461182, 1389.2464600

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3043172, upper bound: 2331.3053624
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A1_B1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3043172, upper bound: 2331.3053624
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -452.0924072, 1769.4573975, -598.5828857, 2343.7353516, -2794.3449707, 2368.0402832
1: -311.6014404, 882.9371948, -413.1731567, 1165.9804688, -1477.5819092, 1296.1102295
2: -175.1000824, 751.4988403, -232.3208618, 991.4859009, -1166.5401611, 983.8196411
3: -220.0936432, 1281.2406006, -292.9925537, 1693.8481445, -1913.9417725, 1574.2331543
4: -301.5140076, 993.7527466, -399.4738159, 1314.0239258, -1615.2548828, 1393.2265625

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3043172, upper bound: 2331.3053624
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3043172, upper bound: 2331.3053624
time: 1.00 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -467.3208313, 1830.5625000, -670.4475098, 2655.2807617, -3122.6015625, 2501.0092773
1: -322.1680908, 912.9301147, -464.9102173, 1309.6298828, -1631.7977295, 1377.8403320
2: -180.9830627, 777.0300903, -260.9217834, 1115.5251465, -1296.5081787, 1037.9517822
3: -227.5158081, 1324.6269531, -328.5774536, 1900.0471191, -2127.5629883, 1653.2043457
4: -311.6451416, 1027.4747314, -448.0463257, 1477.4843750, -1789.1295166, 1475.5207520

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3043172, upper bound: 2331.3053624
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3048864, upper bound: 2331.3053624
time: 1.06 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -452.0924072, 1769.4573975, -693.8747559, 2733.1735840, -3183.5087891, 2463.3320312
1: -311.6014404, 882.9371948, -479.4536743, 1349.8250732, -1661.4265137, 1362.3908691
2: -175.1000824, 751.4988403, -269.0238647, 1148.6254883, -1323.7254639, 1020.5226440
3: -220.0936432, 1281.2406006, -339.0108337, 1959.3713379, -2179.4650879, 1620.2513428
4: -301.5140076, 993.7527466, -462.3641052, 1521.9593506, -1823.4733887, 1456.1168213

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3048864, upper bound: 2331.3053624
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3048864, upper bound: 2331.3053624
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -584.4433594, 2302.2429199, -550.7278442, 2183.4577637, -2767.9011230, 2852.9707031
1: -403.3268127, 1136.8691406, -382.9767456, 1078.0111084, -1481.3377686, 1519.8459473
2: -226.6615906, 968.3230591, -214.8183594, 918.1005249, -1144.7619629, 1183.1413574
3: -284.1044922, 1649.8958740, -270.9054260, 1563.8934326, -1847.9979248, 1920.8012695
4: -389.2089233, 1282.8598633, -368.7893677, 1215.5390625, -1604.7478027, 1651.6491699

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3057703, upper bound: 2331.3074761
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3057703, upper bound: 2331.3074761
time: 1.13 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -620.3299561, 2432.1523438, -533.5703735, 2112.8615723, -2733.1914062, 2964.3315430
1: -427.4905090, 1204.7216797, -370.9188843, 1044.4415283, -1471.9318848, 1575.6406250
2: -240.1427155, 1025.2436523, -208.1604004, 889.4501343, -1129.5928955, 1233.4040527
3: -302.1952820, 1749.5191650, -262.4769592, 1515.1790771, -1817.3742676, 2011.9960938
4: -412.9020386, 1358.6525879, -357.3544617, 1177.5881348, -1590.4902344, 1716.0070801

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3051491, upper bound: 2331.3074761
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3051491, upper bound: 2331.3074761
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -595.3701782, 2346.1115723, -670.4475098, 2655.2807617, -3250.6508789, 3016.5588379
1: -410.8760376, 1158.4462891, -464.9102173, 1309.6298828, -1720.5057373, 1623.3564453
2: -230.8953857, 986.6727905, -260.9217834, 1115.5251465, -1346.4202881, 1247.5944824
3: -289.4539490, 1681.1589355, -328.5774536, 1900.0471191, -2189.5009766, 2009.7363281
4: -396.5617065, 1307.1392822, -448.0463257, 1477.4843750, -1874.0461426, 1755.1855469

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3065012, upper bound: 2331.3078186
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3065012, upper bound: 2331.3078186
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -578.0209351, 2276.1760254, -693.8747559, 2733.1735840, -3309.3669434, 2970.0505371
1: -398.7921143, 1124.7812500, -479.4536743, 1349.8250732, -1748.6170654, 1604.2348633
2: -224.1629028, 958.0428467, -269.0238647, 1148.6254883, -1372.7883301, 1227.0666504
3: -280.9176025, 1632.3837891, -339.0108337, 1959.3713379, -2240.2890625, 1971.3945312
4: -385.0119934, 1269.1036377, -462.3641052, 1521.9593506, -1906.9713135, 1731.4677734

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3065012, upper bound: 2331.3078186
time: 1.69 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3065012, upper bound: 2331.3078186
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -540.3713379, 2141.8784180, -467.3208313, 1830.5625000, -2370.9338379, 2609.1992188
1: -375.7645264, 1057.5384521, -322.1680908, 912.9301147, -1288.6945801, 1379.7065430
2: -210.7732391, 900.6857300, -180.9830627, 777.0300903, -987.8033447, 1081.6687012
3: -265.7885742, 1534.2711182, -227.5158081, 1324.6269531, -1590.4155273, 1761.7868652
4: -361.7720032, 1192.5009766, -311.6451416, 1027.4747314, -1389.2464600, 1504.1461182

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_B1_A1_A1_B1

### Relational analysis result of NS_A2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3053624, upper bound: 2331.3043172
time: 0.78 seconds

## Relational analysis of NS_A2_B1_B1_A1_A1_B2

### Relational analysis result of NS_A2_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3053624, upper bound: 2331.3043172
time: 1.04 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -598.5828857, 2343.7353516, -452.0924072, 1769.4573975, -2368.0402832, 2794.3449707
1: -413.1731567, 1165.9804688, -311.6014404, 882.9371948, -1296.1102295, 1477.5819092
2: -232.3208618, 991.4859009, -175.1000824, 751.4988403, -983.8195801, 1166.5401611
3: -292.9925537, 1693.8481445, -220.0936432, 1281.2406006, -1574.2331543, 1913.9417725
4: -399.4738159, 1314.0239258, -301.5140076, 993.7527466, -1393.2265625, 1615.2548828

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_B1_A1_A2_B1

### Relational analysis result of NS_A2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3053624, upper bound: 2331.3043172
time: 0.98 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2_B2

### Relational analysis result of NS_A2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3053624, upper bound: 2331.3043172
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -670.4475098, 2655.2807617, -467.3208313, 1830.5625000, -2501.0092773, 3122.6015625
1: -464.9102173, 1309.6298828, -322.1680908, 912.9301147, -1377.8403320, 1631.7977295
2: -260.9217834, 1115.5251465, -180.9830627, 777.0300903, -1037.9517822, 1296.5081787
3: -328.5774536, 1900.0471191, -227.5158081, 1324.6269531, -1653.2043457, 2127.5629883
4: -448.0463257, 1477.4843750, -311.6451416, 1027.4747314, -1475.5207520, 1789.1295166

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_B1_A2_A1_B1

### Relational analysis result of NS_A2_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3053624, upper bound: 2331.3048864
time: 1.26 seconds

## Relational analysis of NS_A2_B1_B1_A2_A1_B2

### Relational analysis result of NS_A2_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3053624, upper bound: 2331.3048864
time: 0.94 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -693.8747559, 2733.1735840, -452.0924072, 1769.4573975, -2463.3320312, 3183.5087891
1: -479.4536743, 1349.8250732, -311.6014404, 882.9371948, -1362.3908691, 1661.4265137
2: -269.0238647, 1148.6254883, -175.1000824, 751.4988403, -1020.5226440, 1323.7254639
3: -339.0108337, 1959.3713379, -220.0936432, 1281.2406006, -1620.2513428, 2179.4650879
4: -462.3641052, 1521.9593506, -301.5140076, 993.7527466, -1456.1168213, 1823.4733887

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_B1_A2_A2_B1

### Relational analysis result of NS_A2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3053624, upper bound: 2331.3048864
time: 1.00 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2_B2

### Relational analysis result of NS_A2_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3053624, upper bound: 2331.3048864
time: 0.99 seconds

## BFS NS instance: NS_A2_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -550.7278442, 2183.4577637, -584.4433594, 2302.2429199, -2852.9707031, 2767.9011230
1: -382.9767456, 1078.0111084, -403.3268127, 1136.8691406, -1519.8459473, 1481.3377686
2: -214.8183594, 918.1005249, -226.6615906, 968.3230591, -1183.1413574, 1144.7620850
3: -270.9054260, 1563.8934326, -284.1044922, 1649.8958740, -1920.8012695, 1847.9979248
4: -368.7893677, 1215.5390625, -389.2089233, 1282.8598633, -1651.6491699, 1604.7478027

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3074761, upper bound: 2331.3057703
time: 0.78 seconds

## Relational analysis of NS_A2_B1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3074761, upper bound: 2331.3057703
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -533.5703735, 2112.8615723, -620.3299561, 2432.1523438, -2964.3315430, 2733.1914062
1: -370.9188843, 1044.4415283, -427.4905090, 1204.7216797, -1575.6406250, 1471.9318848
2: -208.1604004, 889.4501343, -240.1427155, 1025.2436523, -1233.4040527, 1129.5928955
3: -262.4769592, 1515.1790771, -302.1952820, 1749.5191650, -2011.9960938, 1817.3742676
4: -357.3544617, 1177.5881348, -412.9020386, 1358.6525879, -1716.0070801, 1590.4902344

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3074761, upper bound: 2331.3057703
time: 0.85 seconds

## Relational analysis of NS_A2_B1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3074761, upper bound: 2331.3057703
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -670.4475098, 2655.2807617, -595.3701782, 2346.1115723, -3016.5588379, 3250.6508789
1: -464.9102173, 1309.6298828, -410.8760376, 1158.4462891, -1623.3564453, 1720.5056152
2: -260.9217834, 1115.5251465, -230.8953857, 986.6727905, -1247.5943604, 1346.4202881
3: -328.5774536, 1900.0471191, -289.4539490, 1681.1589355, -2009.7363281, 2189.5009766
4: -448.0463257, 1477.4843750, -396.5617065, 1307.1392822, -1755.1855469, 1874.0461426

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_B2_A2_A1_B1

### Relational analysis result of NS_A2_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3067938, upper bound: 2331.3064821
time: 0.81 seconds

## Relational analysis of NS_A2_B1_B2_A2_A1_B2

### Relational analysis result of NS_A2_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3067938, upper bound: 2331.3064821
time: 0.99 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -693.8747559, 2733.1735840, -578.0209351, 2276.1760254, -2970.0505371, 3309.3669434
1: -479.4536743, 1349.8250732, -398.7921143, 1124.7812500, -1604.2348633, 1748.6170654
2: -269.0238647, 1148.6254883, -224.1629028, 958.0428467, -1227.0666504, 1372.7883301
3: -339.0108337, 1959.3713379, -280.9176025, 1632.3837891, -1971.3945312, 2240.2890625
4: -462.3641052, 1521.9593506, -385.0119934, 1269.1036377, -1731.4677734, 1906.9713135

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_B2_A2_A2_B1

### Relational analysis result of NS_A2_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3067938, upper bound: 2331.3064821
time: 0.83 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2_B2

### Relational analysis result of NS_A2_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3067938, upper bound: 2331.3064821
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -540.3713379, 2141.8784180, -550.7278442, 2183.4577637, -2723.8291016, 2692.6062012
1: -375.7645264, 1057.5384521, -382.9767456, 1078.0111084, -1453.7756348, 1440.5151367
2: -210.7732391, 900.6857300, -214.8183594, 918.1005249, -1128.8737793, 1115.5041504
3: -265.7885742, 1534.2711182, -270.9054260, 1563.8934326, -1829.6820068, 1805.1765137
4: -361.7720032, 1192.5009766, -368.7893677, 1215.5390625, -1577.3110352, 1561.2902832

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_B1_A1_A1_B1

### Relational analysis result of NS_A2_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3061335, upper bound: 2331.3059101
time: 0.93 seconds

## Relational analysis of NS_A2_B2_B1_A1_A1_B2

### Relational analysis result of NS_A2_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3061335, upper bound: 2331.3059101
time: 0.98 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -598.5828857, 2343.7353516, -533.5703735, 2112.8615723, -2711.4443359, 2875.8701172
1: -413.1731567, 1165.9804688, -370.9188843, 1044.4415283, -1457.6146240, 1536.5750732
2: -232.3208618, 991.4859009, -208.1604004, 889.4501343, -1121.7709961, 1199.3259277
3: -292.9925537, 1693.8481445, -262.4769592, 1515.1790771, -1808.1716309, 1956.2373047
4: -399.4738159, 1314.0239258, -357.3544617, 1177.5881348, -1577.0620117, 1670.9746094

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_B1_A1_A2_B1

### Relational analysis result of NS_A2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3061335, upper bound: 2331.3059101
time: 1.00 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2_B2

### Relational analysis result of NS_A2_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3061335, upper bound: 2331.3059101
time: 0.96 seconds

## BFS NS instance: NS_A2_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -670.4475098, 2655.2807617, -550.7278442, 2183.4577637, -2853.9052734, 3206.0085449
1: -464.9102173, 1309.6298828, -382.9767456, 1078.0111084, -1542.9213867, 1692.6066895
2: -260.9217834, 1115.5251465, -214.8183594, 918.1005249, -1179.0222168, 1330.3435059
3: -328.5774536, 1900.0471191, -270.9054260, 1563.8934326, -1892.4708252, 2170.9526367
4: -448.0463257, 1477.4843750, -368.7893677, 1215.5390625, -1663.5854492, 1846.2736816

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_B1_A2_A1_B1

### Relational analysis result of NS_A2_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3061335, upper bound: 2331.3059420
time: 0.79 seconds

## Relational analysis of NS_A2_B2_B1_A2_A1_B2

### Relational analysis result of NS_A2_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3061335, upper bound: 2331.3059420
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -693.8747559, 2733.1735840, -533.5703735, 2112.8615723, -2806.7363281, 3265.0339355
1: -479.4536743, 1349.8250732, -370.9188843, 1044.4415283, -1523.8952637, 1720.7438965
2: -269.0238647, 1148.6254883, -208.1604004, 889.4501343, -1158.4739990, 1356.7858887
3: -339.0108337, 1959.3713379, -262.4769592, 1515.1790771, -1854.1899414, 2221.8483887
4: -462.3641052, 1521.9593506, -357.3544617, 1177.5881348, -1639.9522705, 1879.3138428

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_B1_A2_A2_B1

### Relational analysis result of NS_A2_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3061335, upper bound: 2331.3059420
time: 1.08 seconds

## Relational analysis of NS_A2_B2_B1_A2_A2_B2

### Relational analysis result of NS_A2_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3061335, upper bound: 2331.3059420
time: 1.08 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -550.7278442, 2183.4577637, -670.4475098, 2655.2807617, -3206.0085449, 2853.9052734
1: -382.9767456, 1078.0111084, -464.9102173, 1309.6298828, -1692.6066895, 1542.9213867
2: -214.8183594, 918.1005249, -260.9217834, 1115.5251465, -1330.3435059, 1179.0220947
3: -270.9054260, 1563.8934326, -328.5774536, 1900.0471191, -2170.9526367, 1892.4708252
4: -368.7893677, 1215.5390625, -448.0463257, 1477.4843750, -1846.2736816, 1663.5854492

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3068882, upper bound: 2331.3060395
time: 1.22 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3068882, upper bound: 2331.3060395
time: 1.19 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -533.5703735, 2112.8615723, -693.8747559, 2733.1735840, -3265.0339355, 2806.7363281
1: -370.9188843, 1044.4415283, -479.4536743, 1349.8250732, -1720.7438965, 1523.8952637
2: -208.1604004, 889.4501343, -269.0238647, 1148.6254883, -1356.7858887, 1158.4738770
3: -262.4769592, 1515.1790771, -339.0108337, 1959.3713379, -2221.8483887, 1854.1899414
4: -357.3544617, 1177.5881348, -462.3641052, 1521.9593506, -1879.3138428, 1639.9522705

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3068882, upper bound: 2331.3060395
time: 0.91 seconds

## Relational analysis of NS_A2_B2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3068882, upper bound: 2331.3060395
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -680.3256226, 2695.0239258, -670.4475098, 2655.2807617, -3335.6062012, 3365.4714355
1: -471.7484131, 1329.2425537, -464.9102173, 1309.6298828, -1781.3779297, 1794.1527100
2: -264.7478638, 1132.2117920, -260.9217834, 1115.5251465, -1380.2728271, 1393.1334229
3: -333.3937683, 1928.4042969, -328.5774536, 1900.0471191, -2233.4406738, 2256.9814453
4: -454.6759949, 1499.5177002, -448.0463257, 1477.4843750, -1932.1604004, 1947.5639648

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3067356, upper bound: 2331.3064821
time: 0.87 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3067356, upper bound: 2331.3064821
time: 1.03 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -658.9322510, 2608.0842285, -693.8747559, 2733.1735840, -3390.6494141, 3301.9589844
1: -456.8302002, 1287.3707275, -479.4536743, 1349.8250732, -1806.6552734, 1766.8244629
2: -256.4273376, 1096.5281982, -269.0238647, 1148.6254883, -1405.0527344, 1365.5520020
3: -322.8622437, 1867.7509766, -339.0108337, 1959.3713379, -2282.2336426, 2206.7617188
4: -440.3402100, 1452.2559814, -462.3641052, 1521.9593506, -1962.2995605, 1914.6201172

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3067356, upper bound: 2331.3064821
time: 1.31 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3067356, upper bound: 2331.3064821
time: 1.19 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.58 seconds
NS_A1_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3111527, upper bound: 2331.3111527
NS_A1_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3111527, upper bound: 2331.3111527
NS_A1_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3111910, upper bound: 2331.3110044
NS_A1_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3110501, upper bound: 2331.3110501
NS_A1_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3061371, upper bound: 2331.3060057
NS_A1_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3082077, upper bound: 2331.3089198
NS_A1_B1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3067910, upper bound: 2331.3034408
NS_A1_B1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3068090, upper bound: 2331.3037879
NS_A1_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3089198, upper bound: 2331.3077128
NS_A1_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3089198, upper bound: 2331.3081594
NS_A1_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3055230, upper bound: 2331.3055230
NS_A1_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3055231, upper bound: 2331.3055231
NS_A1_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3040348, upper bound: 2331.3074064
NS_A1_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3038786, upper bound: 2331.3030736
NS_A1_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3043172, upper bound: 2331.3053624
NS_A1_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3043172, upper bound: 2331.3053624
NS_A1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3043172, upper bound: 2331.3053624
NS_A1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3043172, upper bound: 2331.3053624
NS_A1_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3043172, upper bound: 2331.3053624
NS_A1_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3048864, upper bound: 2331.3053624
NS_A1_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3048864, upper bound: 2331.3053624
NS_A1_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3048864, upper bound: 2331.3053624
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3057703, upper bound: 2331.3074761
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3057703, upper bound: 2331.3074761
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3051491, upper bound: 2331.3074761
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3051491, upper bound: 2331.3074761
NS_A1_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3065012, upper bound: 2331.3078186
NS_A1_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3065012, upper bound: 2331.3078186
NS_A1_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3065012, upper bound: 2331.3078186
NS_A1_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3065012, upper bound: 2331.3078186
NS_A2_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3053624, upper bound: 2331.3043172
NS_A2_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3053624, upper bound: 2331.3043172
NS_A2_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3053624, upper bound: 2331.3043172
NS_A2_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3053624, upper bound: 2331.3043172
NS_A2_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3053624, upper bound: 2331.3048864
NS_A2_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3053624, upper bound: 2331.3048864
NS_A2_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3053624, upper bound: 2331.3048864
NS_A2_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3053624, upper bound: 2331.3048864
NS_A2_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3074761, upper bound: 2331.3057703
NS_A2_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3074761, upper bound: 2331.3057703
NS_A2_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3074761, upper bound: 2331.3057703
NS_A2_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3074761, upper bound: 2331.3057703
NS_A2_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3067938, upper bound: 2331.3064821
NS_A2_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3067938, upper bound: 2331.3064821
NS_A2_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3067938, upper bound: 2331.3064821
NS_A2_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3067938, upper bound: 2331.3064821
NS_A2_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3061335, upper bound: 2331.3059101
NS_A2_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3061335, upper bound: 2331.3059101
NS_A2_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3061335, upper bound: 2331.3059101
NS_A2_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3061335, upper bound: 2331.3059101
NS_A2_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3061335, upper bound: 2331.3059420
NS_A2_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3061335, upper bound: 2331.3059420
NS_A2_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3061335, upper bound: 2331.3059420
NS_A2_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3061335, upper bound: 2331.3059420
NS_A2_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3068882, upper bound: 2331.3060395
NS_A2_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3068882, upper bound: 2331.3060395
NS_A2_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3068882, upper bound: 2331.3060395
NS_A2_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3068882, upper bound: 2331.3060395
NS_A2_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3067356, upper bound: 2331.3064821
NS_A2_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3067356, upper bound: 2331.3064821
NS_A2_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3067356, upper bound: 2331.3064821
NS_A2_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 0, lower bound: -2331.3067356, upper bound: 2331.3064821

## BFS NS instance: NS_A1_B1_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -456.4900513, 1786.6064453, -456.4900513, 1786.6064453, -2243.0964355, 2243.0964355
1: -314.6244202, 891.6278687, -314.6244202, 891.6278687, -1206.2523193, 1206.2523193
2: -176.7469788, 758.8996582, -176.7469788, 758.8996582, -935.6466064, 935.6466064
3: -222.1472168, 1293.7668457, -222.1472168, 1293.7668457, -1515.9140625, 1515.9140625
4: -304.3231812, 1003.4425049, -304.3231812, 1003.4425049, -1307.7653809, 1307.7653809

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_B1_A1_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3086710, upper bound: 2331.3104697
time: 0.76 seconds

## Relational analysis of NS_A1_B1_B1_A1_A1_B1_B2

### Relational analysis result of NS_A1_B1_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3099848, upper bound: 2331.3100622
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -456.4900513, 1786.6064453, -584.4433594, 2302.2429199, -2758.7329102, 2371.0498047
1: -314.6244202, 891.6278687, -403.3268127, 1136.8691406, -1451.4935303, 1294.9544678
2: -176.7469788, 758.8996582, -226.6615906, 968.3230591, -1145.0700684, 985.5612793
3: -222.1472168, 1293.7668457, -284.1044922, 1649.8958740, -1872.0430908, 1577.8713379
4: -304.3231812, 1003.4425049, -389.2089233, 1282.8598633, -1587.1831055, 1392.6510010

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A1

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3109504, upper bound: 2331.3111885
time: 1.17 seconds

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A2

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3109705, upper bound: 2331.3107312
time: 0.94 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -568.3615112, 2238.4965820, -453.9243774, 1776.3489990, -2344.7104492, 2692.4208984
1: -392.2370605, 1105.3776855, -313.2829285, 887.2996216, -1279.5366211, 1418.6605225
2: -220.3873749, 941.5619507, -176.1347504, 755.8003540, -976.1876831, 1117.6966553
3: -276.1569824, 1604.2208252, -221.2277985, 1287.5394287, -1563.6964111, 1825.4484863
4: -378.3522339, 1247.3927002, -302.7840881, 999.6788940, -1378.0311279, 1550.1767578

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A1_A2_B1_B1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3111344, upper bound: 2331.3110044
time: 1.21 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B1_B2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3111344, upper bound: 2331.3110044
time: 0.97 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -572.5570679, 2254.9375000, -494.3878479, 1934.3780518, -2506.9348145, 2749.3249512
1: -395.1222839, 1113.6254883, -341.5354919, 965.7787476, -1360.9010010, 1455.1608887
2: -222.0450439, 948.5439453, -192.3051758, 822.5509033, -1044.5959473, 1140.8486328
3: -278.3341675, 1616.1619873, -241.6322632, 1401.7214355, -1680.0556641, 1857.7941895
4: -381.2427673, 1256.6003418, -330.3079529, 1088.9970703, -1470.2395020, 1586.9083252

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3107312, upper bound: 2331.3110501
time: 1.00 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3107312, upper bound: 2331.3110501
time: 1.24 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -511.7955322, 1987.3428955, -491.5779419, 1920.7878418, -2432.5827637, 2478.9208984
1: -351.3733826, 995.1947021, -338.9739990, 959.6553345, -1311.0286865, 1334.1687012
2: -197.8990631, 845.9482422, -190.7619476, 816.9335938, -1014.8325195, 1036.7102051
3: -249.1619720, 1446.3994141, -239.8054047, 1392.6816406, -1641.8436279, 1686.2048340
4: -340.9003296, 1120.6577148, -328.0542603, 1081.1915283, -1422.0917969, 1448.7119141

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_B1_A2_B2_A1_A1

### Relational analysis result of NS_A1_B1_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3058535, upper bound: 2331.3031490
time: 0.91 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_A1_A2

### Relational analysis result of NS_A1_B1_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3058885, upper bound: 2331.3058215
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -592.7730103, 2312.8642578, -494.3878479, 1934.3780518, -2527.1511230, 2806.9006348
1: -407.0030212, 1150.7774658, -341.5354919, 965.7787476, -1372.7817383, 1492.3129883
2: -229.0003204, 979.1724854, -192.3051758, 822.5509033, -1051.5512695, 1171.4775391
3: -287.7118835, 1670.5908203, -241.6322632, 1401.7214355, -1689.4333496, 1912.2230225
4: -394.0107727, 1296.8674316, -330.3079529, 1088.9970703, -1483.0074463, 1627.1754150

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3081572, upper bound: 2331.3089198
time: 0.71 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3081572, upper bound: 2331.3089198
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -503.8270264, 1969.6516113, -555.6303711, 2159.7888184, -2663.6157227, 2525.2819824
1: -346.8334961, 984.0761108, -381.3075562, 1081.6072998, -1428.4407959, 1365.3836670
2: -195.0213318, 837.5830688, -214.7109070, 919.3721313, -1114.3934326, 1052.2939453
3: -245.1145477, 1427.5999756, -270.3211670, 1571.4353027, -1816.5498047, 1697.9210205
4: -335.6072388, 1107.4798584, -370.1297607, 1217.6490479, -1553.2562256, 1477.6096191

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_B2_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3027454, upper bound: 2331.3017664
time: 0.72 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3061431, upper bound: 2331.3020711
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -502.5316467, 1964.6147461, -552.9077148, 2149.1562500, -2651.6877441, 2517.5219727
1: -345.8869934, 981.5419922, -379.3768921, 1076.3448486, -1422.2316895, 1360.9188232
2: -194.4827576, 835.4318848, -213.6293793, 914.9230347, -1109.4057617, 1049.0612793
3: -244.4199219, 1423.8914795, -268.9591675, 1563.7010498, -1808.1209717, 1692.8505859
4: -334.6977234, 1104.5725098, -368.2798462, 1211.6549072, -1546.3526611, 1472.8522949

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_B2_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3029547, upper bound: 2331.3020647
time: 0.77 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3062460, upper bound: 2331.3024114
time: 1.11 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -446.0828857, 1743.7731934, -560.0328979, 2172.7910156, -2617.8955078, 2303.8054199
1: -307.2007141, 871.0308838, -383.5767517, 1089.1109619, -1396.3115234, 1254.6076660
2: -172.6408234, 741.3766479, -216.2464447, 925.6253052, -1098.2661133, 957.6230469
3: -216.9068451, 1263.8289795, -272.0045166, 1582.2830811, -1799.1898193, 1535.8333740
4: -297.2455444, 980.1732178, -372.8046265, 1225.9661865, -1523.2116699, 1352.9777832

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3059788, upper bound: 2331.3059817
time: 0.78 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3059788, upper bound: 2331.3077128
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -571.4607544, 2249.8928223, -541.5913086, 2109.6647949, -2680.7858887, 2791.4841309
1: -394.0131531, 1111.3277588, -371.4906921, 1054.4707031, -1448.4836426, 1482.8184814
2: -221.5033569, 946.5381470, -209.2172089, 896.5632935, -1118.0666504, 1155.7553711
3: -277.5446472, 1612.7211914, -263.1044006, 1530.9776611, -1808.5223389, 1875.8255615
4: -380.3962097, 1253.8250732, -360.7505493, 1186.7011719, -1567.0974121, 1614.5756836

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_B2_A1_B2_A2_A1

### Relational analysis result of NS_A1_B1_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3086621, upper bound: 2331.3046847
time: 0.85 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2_A2_A2

### Relational analysis result of NS_A1_B1_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3027804, upper bound: 2331.3039836
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -568.5019531, 2211.4204102, -568.5019531, 2211.4204102, -2779.8007812, 2779.8005371
1: -390.3895874, 1107.0008545, -390.3895874, 1107.0008545, -1497.3903809, 1497.3903809
2: -219.7639313, 940.9892578, -219.7639313, 940.9892578, -1160.7531738, 1160.7531738
3: -276.7389832, 1608.2596436, -276.7389832, 1608.2596436, -1884.9986572, 1884.9986572
4: -378.8573914, 1246.3210449, -378.8573914, 1246.3210449, -1625.1784668, 1625.1784668

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_B2_A2_A1_B1_A1

### Relational analysis result of NS_A1_B1_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3017490, upper bound: 2331.2990024
time: 1.30 seconds

## Relational analysis of NS_A1_B1_B2_A2_A1_B1_A2

### Relational analysis result of NS_A1_B1_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.2990582, upper bound: 2331.2989565
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -568.5019531, 2211.4204102, -580.3777466, 2257.6291504, -2825.9960938, 2791.6252441
1: -390.3895874, 1107.0008545, -398.8233032, 1130.7194824, -1521.1091309, 1505.8240967
2: -219.7639313, 940.9892578, -224.6190796, 961.3665161, -1181.1302490, 1165.6082764
3: -276.7389832, 1608.2596436, -282.7142944, 1642.8055420, -1919.5445557, 1890.9738770
4: -378.8573914, 1246.3210449, -387.0067139, 1273.2871094, -1652.1445312, 1633.3276367

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_B2_A2_A1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3017491, upper bound: 2331.2990024
time: 0.96 seconds

## Relational analysis of NS_A1_B1_B2_A2_A1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.2990582, upper bound: 2331.2990247
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -580.3777466, 2257.6291504, -560.6533203, 2180.2458496, -2760.4853516, 2818.1520996
1: -398.8233032, 1130.7194824, -384.9200439, 1091.3408203, -1490.1639404, 1515.6395264
2: -224.6190796, 961.3665161, -216.6894531, 927.6629028, -1152.2818604, 1178.0555420
3: -282.7142944, 1642.8055420, -272.8552856, 1585.6154785, -1868.3297119, 1915.6608887
4: -387.0067139, 1273.2871094, -373.5026855, 1228.6971436, -1615.7036133, 1646.7897949

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_B2_A2_A2_B1_A1

### Relational analysis result of NS_A1_B1_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3038786, upper bound: 2331.3028811
time: 0.78 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2_B1_A2

### Relational analysis result of NS_A1_B1_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3038786, upper bound: 2331.3030736
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -564.6363525, 2195.8774414, -564.4313354, 2195.8374023, -2758.9079590, 2759.8374023
1: -388.1423035, 1100.5201416, -387.8925781, 1104.1451416, -1491.6710205, 1488.4127197
2: -218.6060638, 935.8237915, -218.0135193, 939.1621704, -1157.3293457, 1153.8371582
3: -274.9689636, 1598.7160645, -273.5151672, 1603.0012207, -1877.3883057, 1872.2312012
4: -376.6151733, 1239.2199707, -376.3097839, 1242.2260742, -1618.1337891, 1615.5297852

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_A1

### Relational analysis result of NS_A1_B1_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3038786, upper bound: 2331.3028811
time: 0.79 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3038786, upper bound: 2331.3030736
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -456.4900513, 1786.6064453, -540.3713379, 2141.8784180, -2598.3684082, 2326.9777832
1: -314.6244202, 891.6278687, -375.7645264, 1057.5384521, -1372.1628418, 1267.3923340
2: -176.7469788, 758.8996582, -210.7732391, 900.6857300, -1077.4327393, 969.6729126
3: -222.1472168, 1293.7668457, -265.7885742, 1534.2711182, -1756.4183350, 1559.5554199
4: -304.3231812, 1003.4425049, -361.7720032, 1192.5009766, -1496.8242188, 1365.2142334

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_B1_A1_A1

### Relational analysis result of NS_A1_B2_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3032001, upper bound: 2331.3044247
time: 1.05 seconds

## Relational analysis of NS_A1_B2_A1_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_B1_A1_A1

### Relational analysis result of NS_A1_B2_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3055977, upper bound: 2331.3034815
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_B1_B1_A1_A2

### Relational analysis result of NS_A1_B2_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3056140, upper bound: 2331.3061470
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -525.0745850, 2040.5819092, -540.3713379, 2141.8784180, -2666.9531250, 2580.9531250
1: -360.5375061, 1020.9531860, -375.7645264, 1057.5384521, -1418.0759277, 1396.7177734
2: -203.0270996, 867.7979736, -210.7732391, 900.6857300, -1103.7126465, 1078.5711670
3: -255.7271423, 1483.9025879, -265.7885742, 1534.2711182, -1789.9982910, 1749.6911621
4: -349.7713623, 1149.8022461, -361.7720032, 1192.5009766, -1542.2723389, 1511.5742188

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_B1_A2_A1

### Relational analysis result of NS_A1_B2_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3032001, upper bound: 2331.3044247
time: 1.02 seconds

## Relational analysis of NS_A1_B2_A1_B1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_B1_A2_A1

### Relational analysis result of NS_A1_B2_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.2982625, upper bound: 2331.2967053
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A1_B1_B1_A2_A2

### Relational analysis result of NS_A1_B2_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3044019, upper bound: 2331.3050501
time: 1.00 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -456.4900513, 1786.6064453, -598.5828857, 2343.7353516, -2798.7348633, 2385.1894531
1: -314.6244202, 891.6278687, -413.1731567, 1165.9804688, -1480.6048584, 1304.8010254
2: -176.7469788, 758.8996582, -232.3208618, 991.4859009, -1168.1884766, 991.2205200
3: -222.1472168, 1293.7668457, -292.9925537, 1693.8481445, -1915.9953613, 1586.7593994
4: -304.3231812, 1003.4425049, -399.4738159, 1314.0239258, -1618.0590820, 1402.9161377

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3020835, upper bound: 2331.3052047
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3041456, upper bound: 2331.3052047
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -525.0745850, 2040.5819092, -598.5828857, 2343.7353516, -2866.4011230, 2637.9904785
1: -360.5375061, 1020.9531860, -413.1731567, 1165.9804688, -1526.2491455, 1434.0500488
2: -203.0270996, 867.7979736, -232.3208618, 991.4859009, -1194.3713379, 1100.1188965
3: -255.7271423, 1483.9025879, -292.9925537, 1693.8481445, -1949.4388428, 1776.8951416
4: -349.7713623, 1149.8022461, -399.4738159, 1314.0239258, -1663.2352295, 1549.1156006

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3024136, upper bound: 2331.3024641
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3041352, upper bound: 2331.3025890
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3041456, upper bound: 2331.3052047
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -456.4900513, 1786.6064453, -670.4475098, 2655.2807617, -3111.7707520, 2457.0534668
1: -314.6244202, 891.6278687, -464.9102173, 1309.6298828, -1624.2542725, 1356.5380859
2: -176.7469788, 758.8996582, -260.9217834, 1115.5251465, -1292.2720947, 1019.8214111
3: -222.1472168, 1293.7668457, -328.5774536, 1900.0471191, -2122.1943359, 1622.3442383
4: -304.3231812, 1003.4425049, -448.0463257, 1477.4843750, -1781.8076172, 1451.4885254

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_B1_A1_A1

### Relational analysis result of NS_A1_B2_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3049026, upper bound: 2331.3030279
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_A1_A2

### Relational analysis result of NS_A1_B2_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3049173, upper bound: 2331.3056475
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -525.0745850, 2040.5819092, -670.4475098, 2655.2807617, -3180.3554688, 2711.0292969
1: -360.5375061, 1020.9531860, -464.9102173, 1309.6298828, -1670.1673584, 1485.8634033
2: -203.0270996, 867.7979736, -260.9217834, 1115.5251465, -1318.5522461, 1128.7197266
3: -255.7271423, 1483.9025879, -328.5774536, 1900.0471191, -2155.7741699, 1812.4799805
4: -349.7713623, 1149.8022461, -448.0463257, 1477.4843750, -1827.2557373, 1597.8486328

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_A1

### Relational analysis result of NS_A1_B2_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3049026, upper bound: 2331.3030279
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_A2

### Relational analysis result of NS_A1_B2_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3049173, upper bound: 2331.3056475
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -456.4900513, 1786.6064453, -693.8747559, 2733.1735840, -3187.8986816, 2480.4809570
1: -314.6244202, 891.6278687, -479.4536743, 1349.8250732, -1664.4494629, 1371.0815430
2: -176.7469788, 758.8996582, -269.0238647, 1148.6254883, -1325.3724365, 1027.9233398
3: -222.1472168, 1293.7668457, -339.0108337, 1959.3713379, -2181.5185547, 1632.7775879
4: -304.3231812, 1003.4425049, -462.3641052, 1521.9593506, -1826.2824707, 1465.8065186

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_A1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3046437, upper bound: 2331.3025890
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_A2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3046601, upper bound: 2331.3052047
time: 1.02 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -525.0745850, 2040.5819092, -693.8747559, 2733.1735840, -3255.5649414, 2733.5566406
1: -360.5375061, 1020.9531860, -479.4536743, 1349.8250732, -1710.3625488, 1500.3326416
2: -203.0270996, 867.7979736, -269.0238647, 1148.6254883, -1351.6525879, 1136.8217773
3: -255.7271423, 1483.9025879, -339.0108337, 1959.3713379, -2215.0983887, 1822.9133301
4: -349.7713623, 1149.8022461, -462.3641052, 1521.9593506, -1871.7307129, 1612.0479736

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_A1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3046437, upper bound: 2331.3025890
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_A2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2331.3046601, upper bound: 2331.3052047
time: 1.15 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -584.4433594, 2302.2429199, -540.3713379, 2141.8784180, -2726.3215332, 2842.6142578
1: -403.3268127, 1136.8691406, -375.7645264, 1057.5384521, -1460.8652344, 1512.6336670
2: -226.6615906, 968.3230591, -210.7732391, 900.6857300, -1127.3472900, 1179.0963135
3: -284.1044922, 1649.8958740, -265.7885742, 1534.2711182, -1818.3756104, 1915.6844482
4: -389.2089233, 1282.8598633, -361.7720032, 1192.5009766, -1581.7097168, 1644.6318359

Time for backsubstitution: 1.96 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.92 + 416.30 = 420.22 seconds
