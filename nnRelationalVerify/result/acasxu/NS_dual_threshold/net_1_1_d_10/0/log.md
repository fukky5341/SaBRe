## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_1.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 0)
Time budget: 420 seconds
Split limit: 100
Threshold: 1781.702970027904


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-410.8436279, 1662.2098389, -410.8436279, 1662.2098389, -2073.0534668, 2073.0534668)
1: (-661.0173340, 1839.7109375, -661.0173340, 1839.7109375, -2500.7282715, 2500.7282715)
2: (-493.7767944, 2118.4650879, -493.7767944, 2118.4650879, -2612.2416992, 2612.2416992)
3: (-1061.6599121, 1896.6071777, -1061.6599121, 1896.6071777, -2958.2668457, 2958.2668457)
4: (-851.2732544, 1983.1082764, -851.2732544, 1983.1082764, -2834.3815918, 2834.3815918)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.07 + 1.74 = 2.81 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1781.7386048, upper bound: 1781.7386048

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7386048, upper bound: 1781.7386048
time: 0.57 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7386048, upper bound: 1781.7386048
time: 0.57 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.26 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.26
Output dim: 0, lower bound: -1781.7386048, upper bound: 1781.7386048
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.26
Output dim: 0, lower bound: -1781.7386048, upper bound: 1781.7386048

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -292.2078247, 1185.7346191, -393.4214783, 1591.1026611, -1883.3105469, 1579.1560059
1: -470.8653564, 1312.1569824, -632.9362183, 1761.1723633, -2232.0375977, 1945.0932617
2: -351.4121094, 1511.5142822, -472.6809387, 2027.8708496, -2379.2829590, 1984.1951904
3: -755.9409180, 1351.1541748, -1016.8637695, 1815.5596924, -2571.5004883, 2368.0178223
4: -605.2770386, 1414.5281982, -814.8656006, 1898.5727539, -2503.8498535, 2229.3937988

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7386048, upper bound: 1781.7386048
time: 0.53 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7386048, upper bound: 1781.7386048
time: 0.52 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -407.7161560, 1648.8684082, -408.8272095, 1653.6184082, -2061.3342285, 2057.6955566
1: -656.0112915, 1825.3117676, -657.7915039, 1830.4354248, -2486.4465332, 2483.1032715
2: -490.0357056, 2101.4226074, -491.3654480, 2107.4907227, -2597.5263672, 2592.7880859
3: -1053.7666016, 1881.9282227, -1056.5728760, 1887.1563721, -2940.9228516, 2938.5009766
4: -844.8474731, 1967.4259033, -847.1339111, 1973.0100098, -2817.8571777, 2814.5598145

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7386048, upper bound: 1781.7386048
time: 0.55 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7386048, upper bound: 1781.7386048
time: 0.53 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.25 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -1781.7386048, upper bound: 1781.7386048
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -1781.7386048, upper bound: 1781.7386048
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -1781.7386048, upper bound: 1781.7386048
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -1781.7386048, upper bound: 1781.7386048

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -292.2078247, 1185.7346191, -292.2078247, 1185.7346191, -1477.9423828, 1477.9423828
1: -470.8653564, 1312.1569824, -470.8653564, 1312.1569824, -1783.0222168, 1783.0222168
2: -351.4121094, 1511.5142822, -351.4121094, 1511.5142822, -1862.9263916, 1862.9262695
3: -755.9409180, 1351.1541748, -755.9409180, 1351.1541748, -2107.0952148, 2107.0952148
4: -605.2770386, 1414.5281982, -605.2770386, 1414.5281982, -2019.8050537, 2019.8050537

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7169424, upper bound: 1781.7131983
time: 2.86 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7107045, upper bound: 1781.7107045
time: 0.56 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -292.2078247, 1185.7346191, -407.7161560, 1648.8684082, -1941.0761719, 1593.4508057
1: -470.8653564, 1312.1569824, -656.0112915, 1825.3117676, -2296.1772461, 1968.1682129
2: -351.4121094, 1511.5142822, -490.0357056, 2101.4226074, -2452.8347168, 2001.5500488
3: -755.9409180, 1351.1541748, -1053.7666016, 1881.9282227, -2637.8691406, 2404.9206543
4: -605.2770386, 1414.5281982, -844.8474731, 1967.4259033, -2572.7026367, 2259.3757324

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7169424, upper bound: 1781.7187824
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7107045, upper bound: 1781.7146935
time: 0.74 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -407.7161560, 1648.8684082, -292.2078247, 1185.7346191, -1593.4508057, 1941.0761719
1: -656.0112915, 1825.3117676, -470.8653564, 1312.1569824, -1968.1682129, 2296.1772461
2: -490.0357056, 2101.4226074, -351.4121094, 1511.5142822, -2001.5499268, 2452.8347168
3: -1053.7666016, 1881.9282227, -755.9409180, 1351.1541748, -2404.9206543, 2637.8691406
4: -844.8474731, 1967.4259033, -605.2770386, 1414.5281982, -2259.3757324, 2572.7026367

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7187824, upper bound: 1781.7222047
time: 0.51 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7146935, upper bound: 1781.7124563
time: 0.67 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -407.7161560, 1648.8684082, -407.7161560, 1648.8684082, -2056.5844727, 2056.5844727
1: -656.0112915, 1825.3117676, -656.0112915, 1825.3117676, -2481.3229980, 2481.3229980
2: -490.0357056, 2101.4226074, -490.0357056, 2101.4226074, -2591.4582520, 2591.4582520
3: -1053.7666016, 1881.9282227, -1053.7666016, 1881.9282227, -2935.6948242, 2935.6948242
4: -844.8474731, 1967.4259033, -844.8474731, 1967.4259033, -2812.2729492, 2812.2729492

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7169424, upper bound: 1781.7234835
time: 0.53 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7146935, upper bound: 1781.7222150
time: 0.55 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 1.96 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.96
Output dim: 0, lower bound: -1781.7169424, upper bound: 1781.7131983
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1.96
Output dim: 0, lower bound: -1781.7107045, upper bound: 1781.7107045
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.96
Output dim: 0, lower bound: -1781.7169424, upper bound: 1781.7187824
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.96
Output dim: 0, lower bound: -1781.7107045, upper bound: 1781.7146935
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 1.96
Output dim: 0, lower bound: -1781.7187824, upper bound: 1781.7222047
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 1.96
Output dim: 0, lower bound: -1781.7146935, upper bound: 1781.7124563
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.96
Output dim: 0, lower bound: -1781.7169424, upper bound: 1781.7234835
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.96
Output dim: 0, lower bound: -1781.7146935, upper bound: 1781.7222150

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -276.7585144, 1124.4237061, -283.7498779, 1152.0651855, -1428.8234863, 1408.1735840
1: -446.0162659, 1244.2011719, -457.2524109, 1274.8989258, -1720.9151611, 1701.4536133
2: -332.7811279, 1433.4194336, -341.1999207, 1468.6623535, -1801.4433594, 1774.6193848
3: -715.9465942, 1280.3992920, -734.0437012, 1312.3668213, -2028.3134766, 2014.4429932
4: -573.1133423, 1340.8247070, -587.6505737, 1374.0789795, -1947.1923828, 1928.4752197

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7107045, upper bound: 1781.7107045
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7107045, upper bound: 1781.7107045
time: 0.49 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -371.8542480, 1491.4521484, -278.2041626, 1129.1477051, -1501.0019531, 1769.6560059
1: -601.9755249, 1656.0629883, -448.0940857, 1250.0756836, -1852.0511475, 2104.1569824
2: -449.9300232, 1903.8698730, -334.5020752, 1439.4908447, -1889.4207764, 2238.3718262
3: -969.7329712, 1708.7253418, -719.8632812, 1286.4393311, -2256.1721191, 2428.5886230
4: -774.2708130, 1785.2263184, -576.2658081, 1346.3743896, -2120.6452637, 2361.4912109

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7107045, upper bound: 1781.7107045
time: 0.48 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7107045, upper bound: 1781.7107045
time: 0.54 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -276.7585144, 1124.4237061, -398.9743958, 1613.2980957, -1890.0565186, 1523.3980713
1: -446.0162659, 1244.2011719, -641.8886108, 1786.2307129, -2232.2465820, 1886.0897217
2: -332.7811279, 1433.4194336, -479.4462891, 2056.0900879, -2388.8706055, 1912.8657227
3: -715.9465942, 1280.3992920, -1031.2385254, 1841.1234131, -2557.0700684, 2311.6376953
4: -573.1133423, 1340.8247070, -826.5597534, 1924.8814697, -2497.9943848, 2167.3840332

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7124563, upper bound: 1781.7146914
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7124563, upper bound: 1781.7146935
time: 0.52 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -371.8542480, 1491.4521484, -387.0130615, 1565.5378418, -1937.3920898, 1878.4652100
1: -601.9755249, 1656.0629883, -622.6231079, 1733.7949219, -2335.7705078, 2278.6855469
2: -449.9300232, 1903.8698730, -465.2874451, 1995.2250977, -2445.1550293, 2369.1572266
3: -969.7329712, 1708.7253418, -1000.7440186, 1786.6191406, -2755.1552734, 2709.4692383
4: -774.2708130, 1785.2263184, -802.4636230, 1867.0913086, -2641.3620605, 2587.6896973

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7123884, upper bound: 1781.7143726
time: 0.46 seconds

## Relational analysis of NS_A1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7096915, upper bound: 1781.7113736
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7114094, upper bound: 1781.7119772
time: 0.49 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -398.9743958, 1613.2980957, -276.7585144, 1124.4237061, -1523.3980713, 1890.0565186
1: -641.8886108, 1786.2307129, -446.0162659, 1244.2011719, -1886.0897217, 2232.2465820
2: -479.4462891, 2056.0900879, -332.7811279, 1433.4194336, -1912.8657227, 2388.8706055
3: -1031.2385254, 1841.1234131, -715.9465942, 1280.3992920, -2311.6376953, 2557.0700684
4: -826.5597534, 1924.8814697, -573.1133423, 1340.8247070, -2167.3840332, 2497.9943848

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7146914, upper bound: 1781.7124563
time: 0.49 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7146914, upper bound: 1781.7124563
time: 0.53 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -387.0130615, 1565.5378418, -371.8542480, 1491.4521484, -1878.4652100, 1937.3920898
1: -622.6231079, 1733.7949219, -601.9755249, 1656.0629883, -2278.6853027, 2335.7705078
2: -465.2874451, 1995.2250977, -449.9300232, 1903.8698730, -2369.1572266, 2445.1550293
3: -1000.7440186, 1786.6191406, -969.7329712, 1708.7253418, -2709.4692383, 2755.1552734
4: -802.4636230, 1867.0913086, -774.2708130, 1785.2263184, -2587.6896973, 2641.3620605

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_B2_B1

### Relational analysis result of NS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7143726, upper bound: 1781.7123884
time: 0.49 seconds

## Relational analysis of NS_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B2_B1

### Relational analysis result of NS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7113736, upper bound: 1781.7096915
time: 0.46 seconds

## Relational analysis of NS_A2_B1_B2_B2

### Relational analysis result of NS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7119772, upper bound: 1781.7114094
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -392.9279480, 1589.2556152, -398.9743958, 1613.2980957, -2006.2259521, 1988.2298584
1: -632.1720581, 1759.5748291, -641.8886108, 1786.2307129, -2418.4028320, 2401.4633789
2: -472.1139832, 2025.3890381, -479.4462891, 2056.0900879, -2528.2041016, 2504.8344727
3: -1015.6010742, 1813.4028320, -1031.2385254, 1841.1234131, -2856.7241211, 2844.6406250
4: -813.8695679, 1896.0518799, -826.5597534, 1924.8814697, -2738.7507324, 2722.6113281

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7222753, upper bound: 1781.7222150
time: 0.53 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7222753, upper bound: 1781.7222150
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -484.6770325, 1944.3590088, -387.0130615, 1565.5378418, -2050.2148438, 2331.3720703
1: -783.1220703, 2158.2265625, -622.6231079, 1733.7949219, -2516.9169922, 2780.8491211
2: -585.5126953, 2480.8889160, -465.2874451, 1995.2250977, -2580.7377930, 2946.1755371
3: -1261.0256348, 2226.6118164, -1000.7440186, 1786.6191406, -3045.9340820, 3227.3559570
4: -1008.2093506, 2325.2885742, -802.4636230, 1867.0913086, -2875.3007812, 3127.7521973

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7106615, upper bound: 1781.7077342
time: 0.56 seconds

## Relational analysis of NS_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7064166, upper bound: 1781.7064166
time: 0.58 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.00 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.00
Output dim: 0, lower bound: -1781.7107045, upper bound: 1781.7107045
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.00
Output dim: 0, lower bound: -1781.7107045, upper bound: 1781.7107045
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.00
Output dim: 0, lower bound: -1781.7107045, upper bound: 1781.7107045
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.00
Output dim: 0, lower bound: -1781.7107045, upper bound: 1781.7107045
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.00
Output dim: 0, lower bound: -1781.7124563, upper bound: 1781.7146914
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.00
Output dim: 0, lower bound: -1781.7124563, upper bound: 1781.7146935
NS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.00
Output dim: 0, lower bound: -1781.7096915, upper bound: 1781.7113736
NS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.00
Output dim: 0, lower bound: -1781.7114094, upper bound: 1781.7119772
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.00
Output dim: 0, lower bound: -1781.7146914, upper bound: 1781.7124563
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.00
Output dim: 0, lower bound: -1781.7146914, upper bound: 1781.7124563
NS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.00
Output dim: 0, lower bound: -1781.7113736, upper bound: 1781.7096915
NS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.00
Output dim: 0, lower bound: -1781.7119772, upper bound: 1781.7114094
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.00
Output dim: 0, lower bound: -1781.7222753, upper bound: 1781.7222150
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.00
Output dim: 0, lower bound: -1781.7222753, upper bound: 1781.7222150
NS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.00
Output dim: 0, lower bound: -1781.7106615, upper bound: 1781.7077342
NS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.00
Output dim: 0, lower bound: -1781.7064166, upper bound: 1781.7064166

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -276.7585144, 1124.4237061, -276.7585144, 1124.4237061, -1401.1822510, 1401.1822510
1: -446.0162659, 1244.2011719, -446.0162659, 1244.2011719, -1690.2174072, 1690.2174072
2: -332.7811279, 1433.4194336, -332.7811279, 1433.4194336, -1766.2005615, 1766.2005615
3: -715.9465942, 1280.3992920, -715.9465942, 1280.3992920, -1996.3458252, 1996.3458252
4: -573.1133423, 1340.8247070, -573.1133423, 1340.8247070, -1913.9379883, 1913.9379883

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7169099, upper bound: 1781.7131935
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7169424, upper bound: 1781.7131983
time: 0.45 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -276.7585144, 1124.4237061, -371.8542480, 1491.4521484, -1768.2105713, 1496.2779541
1: -446.0162659, 1244.2011719, -601.9755249, 1656.0629883, -2102.0786133, 1846.1766357
2: -332.7811279, 1433.4194336, -449.9300232, 1903.8698730, -2236.6506348, 1883.3493652
3: -715.9465942, 1280.3992920, -969.7329712, 1708.7253418, -2424.6718750, 2250.1320801
4: -573.1133423, 1340.8247070, -774.2708130, 1785.2263184, -2358.3391113, 2115.0954590

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7160617, upper bound: 1781.7114255
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7169424, upper bound: 1781.7131983
time: 0.49 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -371.8542480, 1491.4521484, -276.7513428, 1124.3959961, -1496.2502441, 1768.2032471
1: -601.9755249, 1656.0629883, -446.0045471, 1244.1697998, -1846.1452637, 2102.0673828
2: -449.9300232, 1903.8698730, -332.7723083, 1433.3839111, -1883.3138428, 2236.6420898
3: -969.7329712, 1708.7253418, -715.9276123, 1280.3671875, -2250.1000977, 2424.6528320
4: -774.2708130, 1785.2263184, -573.0985107, 1340.7912598, -2115.0620117, 2358.3239746

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7090151, upper bound: 1781.7100648
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7106707, upper bound: 1781.7106707
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -371.8542480, 1491.4521484, -371.8542480, 1491.4521484, -1863.3063965, 1863.3063965
1: -601.9755249, 1656.0629883, -601.9755249, 1656.0629883, -2258.0385742, 2258.0385742
2: -449.9300232, 1903.8698730, -449.9300232, 1903.8698730, -2353.7998047, 2353.7998047
3: -969.7329712, 1708.7253418, -969.7329712, 1708.7253418, -2678.0043945, 2678.0043945
4: -774.2708130, 1785.2263184, -774.2708130, 1785.2263184, -2559.4968262, 2559.4968262

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7090151, upper bound: 1781.7100648
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7106707, upper bound: 1781.7106707
time: 0.49 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -276.7585144, 1124.4237061, -392.9279480, 1589.2556152, -1866.0140381, 1517.3516846
1: -446.0162659, 1244.2011719, -632.1720581, 1759.5748291, -2205.5908203, 1876.3731689
2: -332.7811279, 1433.4194336, -472.1139832, 2025.3890381, -2358.1696777, 1905.5334473
3: -715.9465942, 1280.3992920, -1015.6010742, 1813.4028320, -2529.3493652, 2296.0000000
4: -573.1133423, 1340.8247070, -813.8695679, 1896.0518799, -2469.1647949, 2154.6940918

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7188393, upper bound: 1781.7151325
time: 0.46 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7187865, upper bound: 1781.7151236
time: 0.51 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -276.7585144, 1124.4237061, -484.6770325, 1944.3590088, -2221.1174316, 1609.1004639
1: -446.0162659, 1244.2011719, -783.1220703, 2158.2265625, -2604.2421875, 2027.3229980
2: -332.7811279, 1433.4194336, -585.5126953, 2480.8889160, -2813.6691895, 2018.9321289
3: -715.9465942, 1280.3992920, -1261.0256348, 2226.6118164, -2942.5583496, 2541.4248047
4: -573.1133423, 1340.8247070, -1008.2093506, 2325.2885742, -2898.4018555, 2349.0334473

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7210576, upper bound: 1781.7180320
time: 0.51 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7187865, upper bound: 1781.7151236
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -369.6397400, 1482.5614014, -386.0656128, 1561.7060547, -1931.3457031, 1868.6269531
1: -598.4256592, 1646.1999512, -621.0949707, 1729.5557861, -2327.9814453, 2267.2949219
2: -447.2619019, 1892.4997559, -464.1473083, 1990.3350830, -2437.5969238, 2356.6469727
3: -964.0110474, 1698.5567627, -998.2819824, 1782.2497559, -2745.0734863, 2696.8388672
4: -769.6611938, 1774.5987549, -800.4888916, 1862.5104980, -2632.1716309, 2575.0874023

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7096915, upper bound: 1781.7113736
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7096915, upper bound: 1781.7113736
time: 0.48 seconds

## BFS NS instance: NS_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -369.2091370, 1480.5931396, -385.6253357, 1559.9018555, -1929.1107178, 1866.2182617
1: -597.6428223, 1644.0800781, -620.3779297, 1727.5705566, -2325.2131348, 2264.4580078
2: -446.7429504, 1890.0368652, -463.6106873, 1988.0216064, -2434.7646484, 2353.6474609
3: -962.7859497, 1696.2437744, -997.1490479, 1780.1970215, -2741.8544922, 2693.3928223
4: -768.7138062, 1772.2023926, -799.5759888, 1860.3562012, -2629.0698242, 2571.7783203

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7114094, upper bound: 1781.7119772
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7114094, upper bound: 1781.7119772
time: 0.49 seconds

## BFS NS instance: NS_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -392.9279480, 1589.2556152, -276.7585144, 1124.4237061, -1517.3516846, 1866.0140381
1: -632.1720581, 1759.5748291, -446.0162659, 1244.2011719, -1876.3731689, 2205.5908203
2: -472.1139832, 2025.3890381, -332.7811279, 1433.4194336, -1905.5334473, 2358.1696777
3: -1015.6010742, 1813.4028320, -715.9465942, 1280.3992920, -2296.0000000, 2529.3493652
4: -813.8695679, 1896.0518799, -573.1133423, 1340.8247070, -2154.6940918, 2469.1650391

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B1_A1_B1

### Relational analysis result of NS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7151325, upper bound: 1781.7188393
time: 0.59 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2

### Relational analysis result of NS_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7151236, upper bound: 1781.7187865
time: 0.61 seconds

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -484.6770325, 1944.3590088, -276.7585144, 1124.4237061, -1609.1004639, 2221.1174316
1: -783.1220703, 2158.2265625, -446.0162659, 1244.2011719, -2027.3229980, 2604.2421875
2: -585.5126953, 2480.8889160, -332.7811279, 1433.4194336, -2018.9321289, 2813.6691895
3: -1261.0256348, 2226.6118164, -715.9465942, 1280.3992920, -2541.4248047, 2942.5583496
4: -1008.2093506, 2325.2885742, -573.1133423, 1340.8247070, -2349.0334473, 2898.4018555

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B1_A2_A1

### Relational analysis result of NS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7180320, upper bound: 1781.7210576
time: 0.48 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2

### Relational analysis result of NS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7151236, upper bound: 1781.7187865
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -386.0656128, 1561.7060547, -369.6397400, 1482.5614014, -1868.6269531, 1931.3457031
1: -621.0949707, 1729.5557861, -598.4256592, 1646.1999512, -2267.2949219, 2327.9814453
2: -464.1473083, 1990.3350830, -447.2619019, 1892.4997559, -2356.6469727, 2437.5969238
3: -998.2819824, 1782.2497559, -964.0110474, 1698.5567627, -2696.8388672, 2745.0734863
4: -800.4888916, 1862.5104980, -769.6611938, 1774.5987549, -2575.0871582, 2632.1716309

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7113736, upper bound: 1781.7096915
time: 0.54 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7113736, upper bound: 1781.7096915
time: 0.45 seconds

## BFS NS instance: NS_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -385.6253357, 1559.9018555, -369.2091370, 1480.5931396, -1866.2182617, 1929.1107178
1: -620.3779297, 1727.5705566, -597.6428223, 1644.0800781, -2264.4580078, 2325.2131348
2: -463.6106873, 1988.0216064, -446.7429504, 1890.0368652, -2353.6474609, 2434.7646484
3: -997.1490479, 1780.1970215, -962.7859497, 1696.2437744, -2693.3928223, 2741.8544922
4: -799.5759888, 1860.3562012, -768.7138062, 1772.2023926, -2571.7783203, 2629.0698242

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7119772, upper bound: 1781.7114094
time: 0.50 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7119772, upper bound: 1781.7114094
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -392.9279480, 1589.2556152, -392.9279480, 1589.2556152, -1982.1834717, 1982.1834717
1: -632.1720581, 1759.5748291, -632.1720581, 1759.5748291, -2391.7468262, 2391.7468262
2: -472.1139832, 2025.3890381, -472.1139832, 2025.3890381, -2497.5029297, 2497.5029297
3: -1015.6010742, 1813.4028320, -1015.6010742, 1813.4028320, -2829.0029297, 2829.0029297
4: -813.8695679, 1896.0518799, -813.8695679, 1896.0518799, -2709.9213867, 2709.9213867

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7136730, upper bound: 1781.7091794
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7142617, upper bound: 1781.7095939
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -392.9279480, 1589.2556152, -484.6770325, 1944.3590088, -2337.2868652, 2073.9326172
1: -632.1720581, 1759.5748291, -783.1220703, 2158.2265625, -2790.3984375, 2542.6967773
2: -472.1139832, 2025.3890381, -585.5126953, 2480.8889160, -2953.0021973, 2610.9016113
3: -1015.6010742, 1813.4028320, -1261.0256348, 2226.6118164, -3242.2126465, 3072.9328613
4: -813.8695679, 1896.0518799, -1008.2093506, 2325.2885742, -3139.1579590, 2904.2607422

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7136117, upper bound: 1781.7088987
time: 0.52 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2

### Relational analysis result of NS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7142617, upper bound: 1781.7095939
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -460.4322510, 1844.2117920, -371.3363342, 1501.3393555, -1961.7714844, 2215.5480957
1: -744.2117920, 2047.9725342, -597.5398560, 1663.0292969, -2407.2412109, 2645.5124512
2: -556.5748291, 2353.6630859, -446.5787048, 1913.7115479, -2470.2863770, 2800.2416992
3: -1198.3090820, 2113.7905273, -960.3691406, 1713.7304688, -2910.3137207, 3074.1591797
4: -958.5090332, 2206.7712402, -770.3449707, 1790.6446533, -2749.1528320, 2977.1159668

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_A1_B1

### Relational analysis result of NS_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7055619, upper bound: 1781.7055619
time: 0.56 seconds

## Relational analysis of NS_A2_B2_A2_A1_B2

### Relational analysis result of NS_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7055619, upper bound: 1781.7062334
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -483.0009155, 1938.7708740, -383.6017761, 1551.9196777, -2034.9206543, 2322.3725586
1: -779.0424194, 2151.9399414, -617.0953369, 1718.6446533, -2497.6870117, 2769.0349121
2: -582.8635864, 2473.3574219, -461.0997925, 1977.7740479, -2560.6376953, 2934.4570312
3: -1254.5689697, 2220.6840820, -991.8349609, 1770.9937744, -3024.9096680, 3212.5190430
4: -1004.4208984, 2317.7651367, -795.2792358, 1850.6654053, -2855.0864258, 3113.0444336

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_A2_B1

### Relational analysis result of NS_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7055619, upper bound: 1781.7055619
time: 0.47 seconds

## Relational analysis of NS_A2_B2_A2_A2_B2

### Relational analysis result of NS_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7055619, upper bound: 1781.7064166
time: 0.52 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.02 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1781.7169099, upper bound: 1781.7131935
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1781.7169424, upper bound: 1781.7131983
NS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1781.7160617, upper bound: 1781.7114255
NS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1781.7169424, upper bound: 1781.7131983
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1781.7090151, upper bound: 1781.7100648
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1781.7106707, upper bound: 1781.7106707
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1781.7090151, upper bound: 1781.7100648
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1781.7106707, upper bound: 1781.7106707
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1781.7188393, upper bound: 1781.7151325
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1781.7187865, upper bound: 1781.7151236
NS_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1781.7210576, upper bound: 1781.7180320
NS_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1781.7187865, upper bound: 1781.7151236
NS_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1781.7096915, upper bound: 1781.7113736
NS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1781.7096915, upper bound: 1781.7113736
NS_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1781.7114094, upper bound: 1781.7119772
NS_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1781.7114094, upper bound: 1781.7119772
NS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1781.7151325, upper bound: 1781.7188393
NS_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1781.7151236, upper bound: 1781.7187865
NS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1781.7180320, upper bound: 1781.7210576
NS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1781.7151236, upper bound: 1781.7187865
NS_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1781.7113736, upper bound: 1781.7096915
NS_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1781.7113736, upper bound: 1781.7096915
NS_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1781.7119772, upper bound: 1781.7114094
NS_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1781.7119772, upper bound: 1781.7114094
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1781.7136730, upper bound: 1781.7091794
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1781.7142617, upper bound: 1781.7095939
NS_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1781.7136117, upper bound: 1781.7088987
NS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1781.7142617, upper bound: 1781.7095939
NS_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1781.7055619, upper bound: 1781.7055619
NS_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1781.7055619, upper bound: 1781.7062334
NS_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1781.7055619, upper bound: 1781.7055619
NS_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -1781.7055619, upper bound: 1781.7064166

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -274.4427185, 1115.0172119, -275.7986450, 1120.5279541, -1394.9704590, 1390.8159180
1: -442.2922974, 1233.8166504, -444.4732666, 1239.8970947, -1682.1890869, 1678.2897949
2: -329.9755554, 1421.4028320, -331.6188660, 1428.4432373, -1758.4188232, 1753.0217285
3: -709.9766235, 1269.6837158, -713.4709473, 1275.9597168, -1985.9362793, 1983.1546631
4: -568.2381592, 1329.5943604, -571.0941772, 1336.1738281, -1904.4118652, 1900.6884766

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7361857, upper bound: 1781.7361857
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7361857, upper bound: 1781.7361857
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -279.3706665, 1134.2976074, -275.4717407, 1119.3128662, -1398.6835938, 1409.7692871
1: -450.4078979, 1255.0859375, -443.9052734, 1238.5007324, -1688.9086914, 1698.9912109
2: -336.0948792, 1446.0902100, -331.2274780, 1426.8967285, -1762.9915771, 1777.3175049
3: -722.8035278, 1291.5921631, -712.5675659, 1274.4674072, -1997.2709961, 2004.1596680
4: -578.5453491, 1352.6071777, -570.4376831, 1334.6643066, -1913.2095947, 1923.0447998

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7361857, upper bound: 1781.7361857
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7361857, upper bound: 1781.7361857
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -275.7986450, 1120.5279541, -369.6397400, 1482.5614014, -1758.3601074, 1490.1673584
1: -444.4732666, 1239.8970947, -598.4256592, 1646.1999512, -2090.6733398, 1838.3227539
2: -331.6188660, 1428.4432373, -447.2619019, 1892.4997559, -2224.1186523, 1875.7050781
3: -713.4709473, 1275.9597168, -964.0110474, 1698.5567627, -2412.0278320, 2239.9702148
4: -571.0941772, 1336.1738281, -769.6611938, 1774.5987549, -2345.6921387, 2105.8349609

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7160231, upper bound: 1781.7114198
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7160231, upper bound: 1781.7114255
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -275.4717407, 1119.3128662, -369.2091370, 1480.5931396, -1756.0649414, 1488.5218506
1: -443.9052734, 1238.5007324, -597.6428223, 1644.0800781, -2087.9853516, 1836.1435547
2: -331.2274780, 1426.8967285, -446.7429504, 1890.0368652, -2221.2644043, 1873.6396484
3: -712.5675659, 1274.4674072, -962.7859497, 1696.2437744, -2408.8112793, 2237.2534180
4: -570.4376831, 1334.6643066, -768.7138062, 1772.2023926, -2342.6401367, 2103.3779297

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7169099, upper bound: 1781.7131935
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7169099, upper bound: 1781.7131983
time: 0.50 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -369.6397400, 1482.5614014, -275.7915344, 1120.5002441, -1490.1397705, 1758.3529053
1: -598.4256592, 1646.1999512, -444.4615784, 1239.8659668, -1838.2916260, 2090.6616211
2: -447.2619019, 1892.4997559, -331.6100769, 1428.4078369, -1875.6696777, 2224.1098633
3: -964.0110474, 1698.5567627, -713.4520264, 1275.9281006, -2239.9387207, 2412.0085449
4: -769.6611938, 1774.5987549, -571.0794067, 1336.1409912, -2105.8022461, 2345.6774902

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7114198, upper bound: 1781.7160231
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7114198, upper bound: 1781.7160617
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -369.2091370, 1480.5931396, -275.4644775, 1119.2844238, -1488.4934082, 1756.0574951
1: -597.6428223, 1644.0800781, -443.8933716, 1238.4688721, -1836.1116943, 2087.9733887
2: -446.7429504, 1890.0368652, -331.2185059, 1426.8605957, -1873.6035156, 2221.2551270
3: -962.7859497, 1696.2437744, -712.5481567, 1274.4348145, -2237.2207031, 2408.7919922
4: -768.7138062, 1772.2023926, -570.4224854, 1334.6307373, -2103.3442383, 2342.6250000

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7131935, upper bound: 1781.7169099
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7131935, upper bound: 1781.7169424
time: 0.49 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -369.6397400, 1482.5614014, -370.9798279, 1487.9405518, -1857.5799561, 1853.5410156
1: -598.4256592, 1646.1999512, -600.5737915, 1652.1674805, -2250.5932617, 2246.7736816
2: -447.2619019, 1892.4997559, -448.8765259, 1899.3789062, -2346.6408691, 2341.3762207
3: -964.0110474, 1698.5567627, -967.4739380, 1704.7098389, -2668.2866211, 2665.5947266
4: -769.6611938, 1774.5987549, -772.4509277, 1781.0291748, -2550.6904297, 2547.0493164

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7084098, upper bound: 1781.7084098
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7084098, upper bound: 1781.7100648
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -369.2091370, 1480.5931396, -370.3942261, 1485.5953369, -1854.8044434, 1850.9873047
1: -597.6428223, 1644.0800781, -599.6156616, 1649.6062012, -2247.2485352, 2243.6958008
2: -446.7429504, 1890.0368652, -448.1739502, 1896.3953857, -2343.1384277, 2338.2109375
3: -962.7859497, 1696.2437744, -965.9517822, 1701.9965820, -2664.4025879, 2661.9497070
4: -768.7138062, 1772.2023926, -771.2288208, 1778.1854248, -2546.8991699, 2543.4311523

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7100648, upper bound: 1781.7090151
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7100648, upper bound: 1781.7106707
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -274.4427185, 1115.0172119, -391.9052734, 1585.1062012, -1859.5488281, 1506.9224854
1: -442.2922974, 1233.8166504, -630.5175171, 1754.9978027, -2197.2900391, 1864.3339844
2: -329.9755554, 1421.4028320, -470.8795471, 2020.0898438, -2350.0654297, 1892.2823486
3: -709.9766235, 1269.6837158, -1012.9478149, 1808.6712646, -2518.6469727, 2282.6315918
4: -568.2381592, 1329.5943604, -811.7330322, 1891.0808105, -2459.3188477, 2141.3269043

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7273553, upper bound: 1781.7250314
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7273553, upper bound: 1781.7250314
time: 0.49 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -279.3706665, 1134.2976074, -391.7229614, 1584.4055176, -1863.7761230, 1526.0205078
1: -450.4078979, 1255.0859375, -630.2178345, 1754.1907959, -2204.5986328, 1885.3037109
2: -336.0948792, 1446.0902100, -470.6567993, 2019.1923828, -2355.2866211, 1916.7470703
3: -722.8035278, 1291.5921631, -1012.4707031, 1807.8409424, -2530.6442871, 2304.0627441
4: -578.5453491, 1352.6071777, -811.3594360, 1890.2391357, -2468.7844238, 2163.9665527

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7273553, upper bound: 1781.7250314
time: 0.45 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7273553, upper bound: 1781.7250314
time: 0.43 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -275.7986450, 1120.5279541, -482.2149048, 1934.4200439, -2210.2180176, 1602.7429199
1: -444.4732666, 1239.8970947, -779.1491699, 2147.2224121, -2591.6958008, 2019.0462646
2: -331.6188660, 1428.4432373, -582.5481567, 2468.1884766, -2799.8073730, 2010.9914551
3: -713.4709473, 1275.9597168, -1254.6140137, 2215.2812500, -2928.7521973, 2530.5734863
4: -571.0941772, 1336.1738281, -1003.0828857, 2313.4118652, -2884.5056152, 2339.2565918

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7187865, upper bound: 1781.7151236
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7187865, upper bound: 1781.7151236
time: 0.51 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -275.4717407, 1119.3128662, -481.1869202, 1930.1661377, -2205.6372070, 1600.4997559
1: -443.9052734, 1238.5007324, -777.5044556, 2142.6093750, -2586.5144043, 2016.0051270
2: -331.2274780, 1426.8967285, -581.3374023, 2462.7590332, -2793.9865723, 2008.2341309
3: -712.5675659, 1274.4674072, -1252.0045166, 2210.4218750, -2922.9892578, 2526.4716797
4: -570.4376831, 1334.6643066, -1000.9742432, 2308.2561035, -2878.6938477, 2335.6384277

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7187865, upper bound: 1781.7151236
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7187865, upper bound: 1781.7151236
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -369.6397400, 1482.5614014, -389.7070618, 1576.4172363, -1946.0567627, 1872.2684326
1: -598.4256592, 1646.1999512, -627.0286255, 1745.3515625, -2343.7773438, 2273.2285156
2: -447.2619019, 1892.4997559, -468.2453918, 2009.0085449, -2456.2705078, 2360.7451172
3: -964.0110474, 1698.5567627, -1007.3569946, 1798.6240234, -2761.6909180, 2705.9138184
4: -769.6611938, 1774.5987549, -807.2169189, 1880.5397949, -2650.2009277, 2581.8151855

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_A1_B1_B1

### Relational analysis result of NS_A1_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7096915, upper bound: 1781.7113736
time: 0.88 seconds

## Relational analysis of NS_A1_B2_A2_A1_B1_B2

### Relational analysis result of NS_A1_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7096915, upper bound: 1781.7113736
time: 0.52 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -369.6397400, 1482.5614014, -483.7004089, 1940.4158936, -2310.0549316, 1966.2617188
1: -598.4256592, 1646.1999512, -781.5459595, 2153.8608398, -2752.2866211, 2427.7458496
2: -447.2619019, 1892.4997559, -584.3368530, 2475.8500977, -2923.1118164, 2476.8366699
3: -964.0110474, 1698.5567627, -1258.4822998, 2222.1166992, -3183.7602539, 2956.0866699
4: -769.6611938, 1774.5987549, -1006.1760254, 2320.5769043, -3090.2380371, 2780.7734375

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7096915, upper bound: 1781.7113736
time: 0.53 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2_B2

### Relational analysis result of NS_A1_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7096915, upper bound: 1781.7113736
time: 0.54 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -369.2091370, 1480.5931396, -389.5740356, 1575.9079590, -1945.1166992, 1870.1672363
1: -597.6428223, 1644.0800781, -626.8071289, 1744.7583008, -2342.4006348, 2270.8872070
2: -446.7429504, 1890.0368652, -468.0818176, 2008.3562012, -2455.0991211, 2358.1186523
3: -962.7859497, 1696.2437744, -1007.0048828, 1798.0125732, -2759.9040527, 2703.2485352
4: -768.7138062, 1772.2023926, -806.9439087, 1879.9300537, -2648.6437988, 2579.1462402

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_A2_B1_B1

### Relational analysis result of NS_A1_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7114094, upper bound: 1781.7119772
time: 0.49 seconds

## Relational analysis of NS_A1_B2_A2_A2_B1_B2

### Relational analysis result of NS_A1_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7114094, upper bound: 1781.7119772
time: 0.48 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -369.2091370, 1480.5931396, -483.3983459, 1939.1885986, -2308.3974609, 1963.9914551
1: -597.6428223, 1644.0800781, -781.0685425, 2152.5383301, -2750.1804199, 2425.1486816
2: -446.7429504, 1890.0368652, -583.9786987, 2474.2829590, -2921.0258789, 2474.0153809
3: -962.7859497, 1696.2437744, -1257.7347412, 2220.7187500, -3181.1933594, 2953.2180176
4: -768.7138062, 1772.2023926, -1005.5548706, 2319.1008301, -3087.8146973, 2777.7573242

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_A2_B2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7114094, upper bound: 1781.7119772
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2_B2

### Relational analysis result of NS_A1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7100648, upper bound: 1781.7119772
time: 0.50 seconds

## BFS NS instance: NS_A2_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -391.9052734, 1585.1062012, -274.4427185, 1115.0172119, -1506.9224854, 1859.5488281
1: -630.5175171, 1754.9978027, -442.2922974, 1233.8166504, -1864.3339844, 2197.2900391
2: -470.8795471, 2020.0898438, -329.9755554, 1421.4028320, -1892.2823486, 2350.0654297
3: -1012.9478149, 1808.6712646, -709.9766235, 1269.6837158, -2282.6315918, 2518.6469727
4: -811.7330322, 1891.0808105, -568.2381592, 1329.5943604, -2141.3266602, 2459.3188477

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7250314, upper bound: 1781.7273553
time: 0.47 seconds

## Relational analysis of NS_A2_B1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7250314, upper bound: 1781.7273553
time: 0.56 seconds

## BFS NS instance: NS_A2_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -391.7229614, 1584.4055176, -279.3706665, 1134.2976074, -1526.0205078, 1863.7761230
1: -630.2178345, 1754.1907959, -450.4078979, 1255.0859375, -1885.3037109, 2204.5986328
2: -470.6567993, 2019.1923828, -336.0948792, 1446.0902100, -1916.7470703, 2355.2866211
3: -1012.4707031, 1807.8409424, -722.8035278, 1291.5921631, -2304.0627441, 2530.6442871
4: -811.3594360, 1890.2391357, -578.5453491, 1352.6071777, -2163.9665527, 2468.7844238

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7250314, upper bound: 1781.7273553
time: 0.53 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7250314, upper bound: 1781.7273553
time: 0.54 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -482.2149048, 1934.4200439, -275.7986450, 1120.5279541, -1602.7429199, 2210.2180176
1: -779.1491699, 2147.2224121, -444.4732666, 1239.8970947, -2019.0462646, 2591.6958008
2: -582.5481567, 2468.1884766, -331.6188660, 1428.4432373, -2010.9914551, 2799.8073730
3: -1254.6140137, 2215.2812500, -713.4709473, 1275.9597168, -2530.5734863, 2928.7521973
4: -1003.0828857, 2313.4118652, -571.0941772, 1336.1738281, -2339.2563477, 2884.5056152

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B1_A2_A1_B1

### Relational analysis result of NS_A2_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7151236, upper bound: 1781.7187865
time: 0.49 seconds

## Relational analysis of NS_A2_B1_B1_A2_A1_B2

### Relational analysis result of NS_A2_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7151236, upper bound: 1781.7187865
time: 0.57 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -481.1869202, 1930.1661377, -275.4717407, 1119.3128662, -1600.4997559, 2205.6372070
1: -777.5044556, 2142.6093750, -443.9052734, 1238.5007324, -2016.0051270, 2586.5144043
2: -581.3374023, 2462.7590332, -331.2274780, 1426.8967285, -2008.2341309, 2793.9865723
3: -1252.0045166, 2210.4218750, -712.5675659, 1274.4674072, -2526.4719238, 2922.9892578
4: -1000.9742432, 2308.2561035, -570.4376831, 1334.6643066, -2335.6384277, 2878.6938477

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B1_A2_A2_B1

### Relational analysis result of NS_A2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7151236, upper bound: 1781.7187865
time: 0.63 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2_B2

### Relational analysis result of NS_A2_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7151236, upper bound: 1781.7187865
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -389.7070618, 1576.4172363, -369.6397400, 1482.5614014, -1872.2684326, 1946.0567627
1: -627.0286255, 1745.3515625, -598.4256592, 1646.1999512, -2273.2285156, 2343.7773438
2: -468.2453918, 2009.0085449, -447.2619019, 1892.4997559, -2360.7451172, 2456.2705078
3: -1007.3569946, 1798.6240234, -964.0110474, 1698.5567627, -2705.9138184, 2761.6909180
4: -807.2169189, 1880.5397949, -769.6611938, 1774.5987549, -2581.8149414, 2650.2009277

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B2_B1_A1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7113736, upper bound: 1781.7096915
time: 0.56 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7113736, upper bound: 1781.7096915
time: 0.49 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -483.7004089, 1940.4158936, -369.6397400, 1482.5614014, -1966.2617188, 2310.0549316
1: -781.5459595, 2153.8608398, -598.4256592, 1646.1999512, -2427.7458496, 2752.2866211
2: -584.3368530, 2475.8500977, -447.2619019, 1892.4997559, -2476.8366699, 2923.1118164
3: -1258.4822998, 2222.1166992, -964.0110474, 1698.5567627, -2956.0866699, 3183.7602539
4: -1006.1760254, 2320.5769043, -769.6611938, 1774.5987549, -2780.7736816, 3090.2380371

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B2_B1_A2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7113736, upper bound: 1781.7096915
time: 0.63 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7113736, upper bound: 1781.7096915
time: 0.48 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -389.5740356, 1575.9079590, -369.2091370, 1480.5931396, -1870.1672363, 1945.1166992
1: -626.8071289, 1744.7583008, -597.6428223, 1644.0800781, -2270.8872070, 2342.4006348
2: -468.0818176, 2008.3562012, -446.7429504, 1890.0368652, -2358.1186523, 2455.0991211
3: -1007.0048828, 1798.0125732, -962.7859497, 1696.2437744, -2703.2485352, 2759.9040527
4: -806.9439087, 1879.9300537, -768.7138062, 1772.2023926, -2579.1462402, 2648.6437988

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B2_B2_A1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7113736, upper bound: 1781.7114094
time: 0.50 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7113736, upper bound: 1781.7111394
time: 0.52 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -483.3983459, 1939.1885986, -369.2091370, 1480.5931396, -1963.9914551, 2308.3977051
1: -781.0685425, 2152.5383301, -597.6428223, 1644.0800781, -2425.1486816, 2750.1804199
2: -583.9786987, 2474.2829590, -446.7429504, 1890.0368652, -2474.0156250, 2921.0258789
3: -1257.7347412, 2220.7187500, -962.7859497, 1696.2437744, -2953.2180176, 3181.1933594
4: -1005.5548706, 2319.1008301, -768.7138062, 1772.2023926, -2777.7573242, 3087.8146973

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B2_B2_A2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7113736, upper bound: 1781.7114094
time: 0.54 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7113736, upper bound: 1781.7111394
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -366.0052185, 1479.9211426, -375.5512695, 1518.5447998, -1884.5500488, 1855.4722900
1: -589.1425781, 1638.8649902, -604.4079590, 1681.5493164, -2270.6918945, 2243.2729492
2: -440.0267944, 1886.3773193, -451.3991394, 1935.4921875, -2375.5190430, 2337.7763672
3: -946.3379517, 1689.0131836, -970.9180298, 1733.0541992, -2679.3920898, 2659.9311523
4: -758.7982178, 1765.4758301, -778.3224487, 1811.6979980, -2570.4958496, 2543.7978516

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7365958, upper bound: 1781.7366292
time: 0.56 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7365958, upper bound: 1781.7369076
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -417.2734680, 1687.4654541, -390.2654724, 1578.5410156, -1995.8143311, 2077.7309570
1: -670.7286987, 1869.1072998, -627.8210449, 1747.6740723, -2418.4028320, 2496.9274902
2: -501.1014709, 2150.9934082, -468.8138733, 2011.6549072, -2512.7563477, 2619.8068848
3: -1078.9874268, 1924.5645752, -1008.6231079, 1801.0903320, -2880.0773926, 2933.1877441
4: -864.3917847, 2012.0499268, -808.2124023, 1883.0920410, -2747.4833984, 2820.2619629

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7365958, upper bound: 1781.7366302
time: 0.51 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7365958, upper bound: 1781.7369696
time: 0.49 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -375.5512695, 1518.5447998, -460.4322510, 1844.2117920, -2219.7631836, 1978.9770508
1: -604.4079590, 1681.5493164, -744.2117920, 2047.9725342, -2652.3803711, 2425.7612305
2: -451.3991394, 1935.4921875, -556.5748291, 2353.6630859, -2805.0622559, 2492.0666504
3: -970.9180298, 1733.0541992, -1198.3090820, 2113.7905273, -3084.7084961, 2929.9125977
4: -778.3224487, 1811.6979980, -958.5090332, 2206.7712402, -2985.0932617, 2770.2060547

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7136117, upper bound: 1781.7088673
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A1_B2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7136117, upper bound: 1781.7088673
time: 0.52 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -390.2654724, 1578.5410156, -483.0009155, 1938.7708740, -2329.0363770, 2061.5419922
1: -627.8210449, 1747.6740723, -779.0424194, 2151.9399414, -2779.7607422, 2526.7165527
2: -468.8138733, 2011.6549072, -582.8635864, 2473.3574219, -2942.1711426, 2594.5185547
3: -1008.6231079, 1801.0903320, -1254.5689697, 2220.6840820, -3229.3068848, 3055.2150879
4: -808.2124023, 1883.0920410, -1004.4208984, 2317.7651367, -3125.9775391, 2887.5129395

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7136701, upper bound: 1781.7091794
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7136701, upper bound: 1781.7095939
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -460.4322510, 1844.2117920, -362.3964233, 1464.8366699, -1925.2687988, 2206.6081543
1: -744.2117920, 2047.9725342, -583.2001953, 1622.7606201, -2366.9724121, 2631.1726074
2: -556.5748291, 2353.6630859, -435.8930664, 1867.3493652, -2423.9238281, 2789.5561523
3: -1198.3090820, 2113.7905273, -937.2927856, 1672.2458496, -2868.9111328, 3051.0832520
4: -958.5090332, 2206.7712402, -752.0034790, 1747.1445312, -2705.6525879, 2958.7741699

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_A1_B1_B1

### Relational analysis result of NS_A2_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7106615, upper bound: 1781.7072515
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A2_A1_B1_B2

### Relational analysis result of NS_A2_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7106615, upper bound: 1781.7072515
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -460.4322510, 1844.2117920, -411.2694702, 1664.0390625, -2124.4711914, 2255.4812012
1: -744.2117920, 2047.9725342, -661.1314697, 1843.4560547, -2587.6679688, 2709.1040039
2: -556.5748291, 2353.6630859, -493.9898682, 2121.2380371, -2677.8122559, 2847.6528320
3: -1198.3090820, 2113.7905273, -1063.9298096, 1897.1109619, -3093.0681152, 3177.7202148
4: -958.5090332, 2206.7712402, -852.2592773, 1983.1912842, -2941.6997070, 3059.0302734

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7106615, upper bound: 1781.7077342
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A2_A1_B2_B2

### Relational analysis result of NS_A2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7106615, upper bound: 1781.7077342
time: 0.50 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -483.0009155, 1938.7708740, -362.3964233, 1464.8366699, -1947.8375244, 2301.1669922
1: -779.0424194, 2151.9399414, -583.2001953, 1622.7606201, -2401.8029785, 2735.1396484
2: -582.8635864, 2473.3574219, -435.8930664, 1867.3493652, -2450.2128906, 2909.2502441
3: -1254.5689697, 2220.6840820, -937.2927856, 1672.2458496, -2926.2536621, 3157.9765625
4: -1004.4208984, 2317.7651367, -752.0034790, 1747.1445312, -2751.5654297, 3069.7683105

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7036141, upper bound: 1781.7037261
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7037854, upper bound: 1781.7037854
time: 0.54 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -483.0009155, 1938.7708740, -411.8975830, 1666.5137939, -2149.5146484, 2350.6682129
1: -779.0424194, 2151.9399414, -662.1156006, 1846.2097168, -2625.2521973, 2814.0554199
2: -582.8635864, 2473.3574219, -494.7460022, 2124.3984375, -2707.2619629, 2968.1032715
3: -1254.5689697, 2220.6840820, -1065.5502930, 1899.9202881, -3153.5915527, 3286.2341309
4: -1004.4208984, 2317.7651367, -853.5563965, 1986.1309814, -2990.5517578, 3171.3215332

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_A2_B2_B1

### Relational analysis result of NS_A2_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7055619, upper bound: 1781.7064166
time: 0.50 seconds

## Relational analysis of NS_A2_B2_A2_A2_B2_B2

### Relational analysis result of NS_A2_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7055619, upper bound: 1781.7064166
time: 0.59 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 1.97 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7361857, upper bound: 1781.7361857
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7361857, upper bound: 1781.7361857
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7361857, upper bound: 1781.7361857
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7361857, upper bound: 1781.7361857
NS_A1_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7160231, upper bound: 1781.7114198
NS_A1_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7160231, upper bound: 1781.7114255
NS_A1_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7169099, upper bound: 1781.7131935
NS_A1_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7169099, upper bound: 1781.7131983
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7114198, upper bound: 1781.7160231
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7114198, upper bound: 1781.7160617
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7131935, upper bound: 1781.7169099
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7131935, upper bound: 1781.7169424
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7084098, upper bound: 1781.7084098
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7084098, upper bound: 1781.7100648
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7100648, upper bound: 1781.7090151
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7100648, upper bound: 1781.7106707
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7273553, upper bound: 1781.7250314
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7273553, upper bound: 1781.7250314
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7273553, upper bound: 1781.7250314
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7273553, upper bound: 1781.7250314
NS_A1_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7187865, upper bound: 1781.7151236
NS_A1_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7187865, upper bound: 1781.7151236
NS_A1_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7187865, upper bound: 1781.7151236
NS_A1_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7187865, upper bound: 1781.7151236
NS_A1_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7096915, upper bound: 1781.7113736
NS_A1_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7096915, upper bound: 1781.7113736
NS_A1_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7096915, upper bound: 1781.7113736
NS_A1_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7096915, upper bound: 1781.7113736
NS_A1_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7114094, upper bound: 1781.7119772
NS_A1_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7114094, upper bound: 1781.7119772
NS_A1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7114094, upper bound: 1781.7119772
NS_A1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7100648, upper bound: 1781.7119772
NS_A2_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7250314, upper bound: 1781.7273553
NS_A2_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7250314, upper bound: 1781.7273553
NS_A2_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7250314, upper bound: 1781.7273553
NS_A2_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7250314, upper bound: 1781.7273553
NS_A2_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7151236, upper bound: 1781.7187865
NS_A2_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7151236, upper bound: 1781.7187865
NS_A2_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7151236, upper bound: 1781.7187865
NS_A2_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7151236, upper bound: 1781.7187865
NS_A2_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7113736, upper bound: 1781.7096915
NS_A2_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7113736, upper bound: 1781.7096915
NS_A2_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7113736, upper bound: 1781.7096915
NS_A2_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7113736, upper bound: 1781.7096915
NS_A2_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7113736, upper bound: 1781.7114094
NS_A2_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7113736, upper bound: 1781.7111394
NS_A2_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7113736, upper bound: 1781.7114094
NS_A2_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7113736, upper bound: 1781.7111394
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7365958, upper bound: 1781.7366292
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7365958, upper bound: 1781.7369076
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7365958, upper bound: 1781.7366302
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7365958, upper bound: 1781.7369696
NS_A2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7136117, upper bound: 1781.7088673
NS_A2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7136117, upper bound: 1781.7088673
NS_A2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7136701, upper bound: 1781.7091794
NS_A2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7136701, upper bound: 1781.7095939
NS_A2_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7106615, upper bound: 1781.7072515
NS_A2_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7106615, upper bound: 1781.7072515
NS_A2_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7106615, upper bound: 1781.7077342
NS_A2_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7106615, upper bound: 1781.7077342
NS_A2_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7036141, upper bound: 1781.7037261
NS_A2_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7037854, upper bound: 1781.7037854
NS_A2_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7055619, upper bound: 1781.7064166
NS_A2_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 1.97
Output dim: 0, lower bound: -1781.7055619, upper bound: 1781.7064166

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -274.4427185, 1115.0172119, -274.4427185, 1115.0172119, -1389.4598389, 1389.4598389
1: -442.2922974, 1233.8166504, -442.2922974, 1233.8166504, -1676.1086426, 1676.1086426
2: -329.9755554, 1421.4028320, -329.9755554, 1421.4028320, -1751.3784180, 1751.3784180
3: -709.9766235, 1269.6837158, -709.9766235, 1269.6837158, -1979.6600342, 1979.6600342
4: -568.2381592, 1329.5943604, -568.2381592, 1329.5943604, -1897.8323975, 1897.8323975

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7347025, upper bound: 1781.7324269
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7307421, upper bound: 1781.7307421
time: 0.50 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -274.4427185, 1115.0172119, -279.3706665, 1134.2976074, -1408.7403564, 1394.3879395
1: -442.2922974, 1233.8166504, -450.4078979, 1255.0859375, -1697.3780518, 1684.2246094
2: -329.9755554, 1421.4028320, -336.0948792, 1446.0902100, -1776.0656738, 1757.4976807
3: -709.9766235, 1269.6837158, -722.8035278, 1291.5921631, -2001.5687256, 1992.4871826
4: -568.2381592, 1329.5943604, -578.5453491, 1352.6071777, -1920.8453369, 1908.1396484

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7347025, upper bound: 1781.7332929
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7307421, upper bound: 1781.7314321
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -279.3706665, 1134.2976074, -274.4427185, 1115.0172119, -1394.3879395, 1408.7403564
1: -450.4078979, 1255.0859375, -442.2922974, 1233.8166504, -1684.2246094, 1697.3780518
2: -336.0948792, 1446.0902100, -329.9755554, 1421.4028320, -1757.4976807, 1776.0656738
3: -722.8035278, 1291.5921631, -709.9766235, 1269.6837158, -1992.4871826, 2001.5687256
4: -578.5453491, 1352.6071777, -568.2381592, 1329.5943604, -1908.1396484, 1920.8453369

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7332929, upper bound: 1781.7346340
time: 0.48 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7314321, upper bound: 1781.7310731
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -279.3706665, 1134.2976074, -279.3706665, 1134.2976074, -1413.6682129, 1413.6682129
1: -450.4078979, 1255.0859375, -450.4078979, 1255.0859375, -1705.4938965, 1705.4938965
2: -336.0948792, 1446.0902100, -336.0948792, 1446.0902100, -1782.1850586, 1782.1850586
3: -722.8035278, 1291.5921631, -722.8035278, 1291.5921631, -2014.3957520, 2014.3957520
4: -578.5453491, 1352.6071777, -578.5453491, 1352.6071777, -1931.1525879, 1931.1525879

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7340968, upper bound: 1781.7331355
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7314321, upper bound: 1781.7322615
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -274.4427185, 1115.0172119, -369.6397400, 1482.5614014, -1757.0040283, 1484.6567383
1: -442.2922974, 1233.8166504, -598.4256592, 1646.1999512, -2088.4921875, 1832.2423096
2: -329.9755554, 1421.4028320, -447.2619019, 1892.4997559, -2222.4750977, 1868.6646729
3: -709.9766235, 1269.6837158, -964.0110474, 1698.5567627, -2408.5327148, 2233.6948242
4: -568.2381592, 1329.5943604, -769.6611938, 1774.5987549, -2342.8369141, 2099.2556152

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 36

## BFS NS instance: NS_A1_B1_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -279.3706665, 1134.2976074, -369.6397400, 1482.5614014, -1761.9321289, 1503.9373779
1: -450.4078979, 1255.0859375, -598.4256592, 1646.1999512, -2096.6079102, 1853.5115967
2: -336.0948792, 1446.0902100, -447.2619019, 1892.4997559, -2228.5947266, 1893.3519287
3: -722.8035278, 1291.5921631, -964.0110474, 1698.5567627, -2421.3603516, 2255.6030273
4: -578.5453491, 1352.6071777, -769.6611938, 1774.5987549, -2353.1433105, 2122.2683105

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 36

## BFS NS instance: NS_A1_B1_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -274.4427185, 1115.0172119, -369.2091370, 1480.5931396, -1755.0357666, 1484.2263184
1: -442.2922974, 1233.8166504, -597.6428223, 1644.0800781, -2086.3723145, 1831.4594727
2: -329.9755554, 1421.4028320, -446.7429504, 1890.0368652, -2220.0122070, 1868.1457520
3: -709.9766235, 1269.6837158, -962.7859497, 1696.2437744, -2406.2202148, 2232.4697266
4: -568.2381592, 1329.5943604, -768.7138062, 1772.2023926, -2340.4404297, 2098.3081055

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 36

## BFS NS instance: NS_A1_B1_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -279.3706665, 1134.2976074, -369.2091370, 1480.5931396, -1759.9638672, 1503.5067139
1: -450.4078979, 1255.0859375, -597.6428223, 1644.0800781, -2094.4880371, 1852.7287598
2: -336.0948792, 1446.0902100, -446.7429504, 1890.0368652, -2226.1318359, 1892.8331299
3: -722.8035278, 1291.5921631, -962.7859497, 1696.2437744, -2419.0473633, 2254.3781738
4: -578.5453491, 1352.6071777, -768.7138062, 1772.2023926, -2350.7475586, 2121.3210449

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 36

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -369.6397400, 1482.5614014, -274.4358215, 1114.9897461, -1484.6293945, 1756.9970703
1: -598.4256592, 1646.1999512, -442.2808838, 1233.7861328, -1832.2116699, 2088.4809570
2: -447.2619019, 1892.4997559, -329.9669800, 1421.3682861, -1868.6301270, 2222.4665527
3: -964.0110474, 1698.5567627, -709.9579468, 1269.6525879, -2233.6635742, 2408.5141602
4: -769.6611938, 1774.5987549, -568.2235718, 1329.5618896, -2099.2231445, 2342.8220215

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 36

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -369.6397400, 1482.5614014, -279.3627014, 1134.2663574, -1503.9056396, 1761.9240723
1: -598.4256592, 1646.1999512, -450.3945923, 1255.0509033, -1853.4765625, 2096.5944824
2: -447.2619019, 1892.4997559, -336.0849915, 1446.0505371, -1893.3122559, 2228.5847168
3: -964.0110474, 1698.5567627, -722.7821045, 1291.5561523, -2255.5668945, 2421.3388672
4: -769.6611938, 1774.5987549, -578.5285645, 1352.5700684, -2122.2312012, 2353.1271973

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 36

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -369.2091370, 1480.5931396, -274.4358215, 1114.9897461, -1484.1988525, 1755.0289307
1: -597.6428223, 1644.0800781, -442.2808838, 1233.7861328, -1831.4289551, 2086.3608398
2: -446.7429504, 1890.0368652, -329.9669800, 1421.3682861, -1868.1112061, 2220.0036621
3: -962.7859497, 1696.2437744, -709.9579468, 1269.6525879, -2232.4384766, 2406.2016602
4: -768.7138062, 1772.2023926, -568.2235718, 1329.5618896, -2098.2756348, 2340.4260254

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 36

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -369.2091370, 1480.5931396, -279.3627014, 1134.2663574, -1503.4753418, 1759.9558105
1: -597.6428223, 1644.0800781, -450.3945923, 1255.0509033, -1852.6937256, 2094.4746094
2: -446.7429504, 1890.0368652, -336.0849915, 1446.0505371, -1892.7934570, 2226.1218262
3: -962.7859497, 1696.2437744, -722.7821045, 1291.5561523, -2254.3420410, 2419.0258789
4: -768.7138062, 1772.2023926, -578.5285645, 1352.5700684, -2121.2839355, 2350.7309570

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 36

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -369.6397400, 1482.5614014, -369.6397400, 1482.5614014, -1852.2009277, 1852.2009277
1: -598.4256592, 1646.1999512, -598.4256592, 1646.1999512, -2244.6254883, 2244.6254883
2: -447.2619019, 1892.4997559, -447.2619019, 1892.4997559, -2339.7617188, 2339.7617188
3: -964.0110474, 1698.5567627, -964.0110474, 1698.5567627, -2662.1411133, 2662.1411133
4: -769.6611938, 1774.5987549, -769.6611938, 1774.5987549, -2544.2595215, 2544.2592773

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -369.6397400, 1482.5614014, -369.2091370, 1480.5931396, -1850.2326660, 1851.7705078
1: -598.4256592, 1646.1999512, -597.6428223, 1644.0800781, -2242.5058594, 2243.8425293
2: -447.2619019, 1892.4997559, -446.7429504, 1890.0368652, -2337.2988281, 2339.2426758
3: -964.0110474, 1698.5567627, -962.7859497, 1696.2437744, -2660.0251465, 2660.9721680
4: -769.6611938, 1774.5987549, -768.7138062, 1772.2023926, -2541.8635254, 2543.3122559

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -369.2091370, 1480.5931396, -369.6397400, 1482.5614014, -1851.7705078, 1850.2326660
1: -597.6428223, 1644.0800781, -598.4256592, 1646.1999512, -2243.8422852, 2242.5058594
2: -446.7429504, 1890.0368652, -447.2619019, 1892.4997559, -2339.2426758, 2337.2988281
3: -962.7859497, 1696.2437744, -964.0110474, 1698.5567627, -2660.9721680, 2660.0251465
4: -768.7138062, 1772.2023926, -769.6611938, 1774.5987549, -2543.3120117, 2541.8635254

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -369.2091370, 1480.5931396, -369.2091370, 1480.5931396, -1849.8022461, 1849.8022461
1: -597.6428223, 1644.0800781, -597.6428223, 1644.0800781, -2241.7226562, 2241.7229004
2: -446.7429504, 1890.0368652, -446.7429504, 1890.0368652, -2336.7797852, 2336.7797852
3: -962.7859497, 1696.2437744, -962.7859497, 1696.2437744, -2658.8562012, 2658.8562012
4: -768.7138062, 1772.2023926, -768.7138062, 1772.2023926, -2540.9162598, 2540.9162598

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -274.4427185, 1115.0172119, -390.3378906, 1578.7467041, -1853.1893311, 1505.3551025
1: -442.2922974, 1233.8166504, -627.9822998, 1747.9838867, -2190.2761230, 1861.7988281
2: -329.9755554, 1421.4028320, -468.9877930, 2011.9674072, -2341.9428711, 1890.3906250
3: -709.9766235, 1269.6837158, -1008.8870239, 1801.4213867, -2511.3974609, 2278.5705566
4: -568.2381592, 1329.5943604, -808.4592896, 1883.4610596, -2451.6989746, 2138.0532227

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7266354, upper bound: 1781.7242379
time: 0.50 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7264707, upper bound: 1781.7241928
time: 0.50 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -274.4427185, 1115.0172119, -388.5496521, 1572.0577393, -1846.5003662, 1503.5668945
1: -442.2922974, 1233.8166504, -625.2743530, 1740.4183350, -2182.7106934, 1859.0909424
2: -329.9755554, 1421.4028320, -466.9485779, 2003.2802734, -2333.2558594, 1888.3513184
3: -709.9766235, 1269.6837158, -1004.1314087, 1793.6044922, -2503.5808105, 2273.8151855
4: -568.2381592, 1329.5943604, -804.7737427, 1875.3979492, -2443.6359863, 2134.3681641

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 28

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -279.3706665, 1134.2976074, -390.3378906, 1578.7467041, -1858.1174316, 1524.6354980
1: -450.4078979, 1255.0859375, -627.9822998, 1747.9838867, -2198.3918457, 1883.0682373
2: -336.0948792, 1446.0902100, -468.9877930, 2011.9674072, -2348.0622559, 1915.0780029
3: -722.8035278, 1291.5921631, -1008.8870239, 1801.4213867, -2524.2248535, 2300.4792480
4: -578.5453491, 1352.6071777, -808.4592896, 1883.4610596, -2462.0058594, 2161.0664062

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 28

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -279.3706665, 1134.2976074, -388.5496521, 1572.0577393, -1851.4284668, 1522.8472900
1: -450.4078979, 1255.0859375, -625.2743530, 1740.4183350, -2190.8261719, 1880.3603516
2: -336.0948792, 1446.0902100, -466.9485779, 2003.2802734, -2339.3752441, 1913.0385742
3: -722.8035278, 1291.5921631, -1004.1314087, 1793.6044922, -2516.4079590, 2295.7236328
4: -578.5453491, 1352.6071777, -804.7737427, 1875.3979492, -2453.9428711, 2157.3808594

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 28

## BFS NS instance: NS_A1_B2_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -274.4427185, 1115.0172119, -482.2149048, 1934.4200439, -2208.8625488, 1597.2321777
1: -442.2922974, 1233.8166504, -779.1491699, 2147.2224121, -2589.5146484, 2012.9658203
2: -329.9755554, 1421.4028320, -582.5481567, 2468.1884766, -2798.1635742, 2003.9509277
3: -709.9766235, 1269.6837158, -1254.6140137, 2215.2812500, -2925.2573242, 2524.2976074
4: -568.2381592, 1329.5943604, -1003.0828857, 2313.4118652, -2881.6499023, 2332.6767578

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7190473, upper bound: 1781.7158082
time: 0.48 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7207544, upper bound: 1781.7173608
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -279.3706665, 1134.2976074, -482.2149048, 1934.4200439, -2213.7907715, 1616.5124512
1: -450.4078979, 1255.0859375, -779.1491699, 2147.2224121, -2597.6303711, 2034.2351074
2: -336.0948792, 1446.0902100, -582.5481567, 2468.1884766, -2804.2832031, 2028.6383057
3: -722.8035278, 1291.5921631, -1254.6140137, 2215.2812500, -2938.0847168, 2546.2060547
4: -578.5453491, 1352.6071777, -1003.0828857, 2313.4118652, -2891.9567871, 2355.6899414

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7190473, upper bound: 1781.7158082
time: 0.50 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7207544, upper bound: 1781.7173608
time: 0.50 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -274.4427185, 1115.0172119, -481.1869202, 1930.1661377, -2204.6083984, 1596.2041016
1: -442.2922974, 1233.8166504, -777.5044556, 2142.6093750, -2584.9016113, 2011.3210449
2: -329.9755554, 1421.4028320, -581.3374023, 2462.7590332, -2792.7341309, 2002.7402344
3: -709.9766235, 1269.6837158, -1252.0045166, 2210.4218750, -2920.3977051, 2521.6882324
4: -568.2381592, 1329.5943604, -1000.9742432, 2308.2561035, -2876.4941406, 2330.5686035

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7169148, upper bound: 1781.7139555
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7184688, upper bound: 1781.7147153
time: 0.54 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -279.3706665, 1134.2976074, -481.1869202, 1930.1661377, -2209.5366211, 1615.4844971
1: -450.4078979, 1255.0859375, -777.5044556, 2142.6093750, -2593.0166016, 2032.5903320
2: -336.0948792, 1446.0902100, -581.3374023, 2462.7590332, -2798.8537598, 2027.4276123
3: -722.8035278, 1291.5921631, -1252.0045166, 2210.4218750, -2933.2253418, 2543.5966797
4: -578.5453491, 1352.6071777, -1000.9742432, 2308.2561035, -2886.8012695, 2353.5815430

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7169148, upper bound: 1781.7139581
time: 0.50 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7184688, upper bound: 1781.7147153
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -369.6397400, 1482.5614014, -388.1438904, 1570.0700684, -1939.7097168, 1870.7052002
1: -598.4256592, 1646.1999512, -624.4998779, 1738.3515625, -2336.7773438, 2270.6997070
2: -447.2619019, 1892.4997559, -466.3586121, 2000.9025879, -2448.1643066, 2358.8583984
3: -964.0110474, 1698.5567627, -1003.3052979, 1791.3928223, -2754.4746094, 2701.8620605
4: -769.6611938, 1774.5987549, -803.9530029, 1872.9388428, -2642.5998535, 2578.5512695

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 36

## BFS NS instance: NS_A1_B2_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -369.6397400, 1482.5614014, -386.6103516, 1564.3378906, -1933.9774170, 1869.1717529
1: -598.4256592, 1646.1999512, -622.1891479, 1731.8579102, -2330.2836914, 2268.3891602
2: -447.2619019, 1892.4997559, -464.6183167, 1993.4416504, -2440.7036133, 2357.1181641
3: -964.0110474, 1698.5567627, -999.1990967, 1784.6806641, -2747.9895020, 2697.7558594
4: -769.6611938, 1774.5987549, -800.7770386, 1866.0429688, -2635.7041016, 2575.3752441

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 36

## BFS NS instance: NS_A1_B2_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -369.6397400, 1482.5614014, -482.2149048, 1934.4200439, -2304.0595703, 1964.7763672
1: -598.4256592, 1646.1999512, -779.1491699, 2147.2224121, -2745.6479492, 2425.3491211
2: -447.2619019, 1892.4997559, -582.5481567, 2468.1884766, -2915.4504395, 2475.0478516
3: -964.0110474, 1698.5567627, -1254.6140137, 2215.2812500, -3176.9301758, 2952.2219238
4: -769.6611938, 1774.5987549, -1003.0828857, 2313.4118652, -3083.0729980, 2777.6806641

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 35

## BFS NS instance: NS_A1_B2_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -369.6397400, 1482.5614014, -481.1869202, 1930.1661377, -2299.8054199, 1963.7482910
1: -598.4256592, 1646.1999512, -777.5044556, 2142.6093750, -2741.0346680, 2423.7043457
2: -447.2619019, 1892.4997559, -581.3374023, 2462.7590332, -2910.0209961, 2473.8371582
3: -964.0110474, 1698.5567627, -1252.0045166, 2210.4218750, -3172.2661133, 2949.6840820
4: -769.6611938, 1774.5987549, -1000.9742432, 2308.2561035, -3077.9172363, 2775.5725098

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 35

## BFS NS instance: NS_A1_B2_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -369.2091370, 1480.5931396, -388.1438904, 1570.0700684, -1939.2791748, 1868.7369385
1: -597.6428223, 1644.0800781, -624.4998779, 1738.3515625, -2335.9941406, 2268.5800781
2: -446.7429504, 1890.0368652, -466.3586121, 2000.9025879, -2447.6455078, 2356.3955078
3: -962.7859497, 1696.2437744, -1003.3052979, 1791.3928223, -2753.3059082, 2699.5490723
4: -768.7138062, 1772.2023926, -803.9530029, 1872.9388428, -2641.6525879, 2576.1552734

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 36

## BFS NS instance: NS_A1_B2_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -369.2091370, 1480.5931396, -386.6103516, 1564.3378906, -1933.5469971, 1867.2034912
1: -597.6428223, 1644.0800781, -622.1891479, 1731.8579102, -2329.5004883, 2266.2692871
2: -446.7429504, 1890.0368652, -464.6183167, 1993.4416504, -2440.1845703, 2354.6550293
3: -962.7859497, 1696.2437744, -999.1990967, 1784.6806641, -2746.8208008, 2695.4428711
4: -768.7138062, 1772.2023926, -800.7770386, 1866.0429688, -2634.7565918, 2572.9794922

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 36

## BFS NS instance: NS_A1_B2_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -369.2091370, 1480.5931396, -482.2149048, 1934.4200439, -2303.6291504, 1962.8081055
1: -597.6428223, 1644.0800781, -779.1491699, 2147.2224121, -2744.8649902, 2423.2292480
2: -446.7429504, 1890.0368652, -582.5481567, 2468.1884766, -2914.9313965, 2472.5849609
3: -962.7859497, 1696.2437744, -1254.6140137, 2215.2812500, -3175.7614746, 2950.1059570
4: -768.7138062, 1772.2023926, -1003.0828857, 2313.4118652, -3082.1254883, 2775.2851562

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## BFS NS instance: NS_A1_B2_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -369.2091370, 1480.5931396, -481.1869202, 1930.1661377, -2299.3747559, 1961.7800293
1: -597.6428223, 1644.0800781, -777.5044556, 2142.6093750, -2740.2512207, 2421.5844727
2: -446.7429504, 1890.0368652, -581.3374023, 2462.7590332, -2909.5019531, 2471.3742676
3: -962.7859497, 1696.2437744, -1252.0045166, 2210.4218750, -3171.0974121, 2947.5681152
4: -768.7138062, 1772.2023926, -1000.9742432, 2308.2561035, -3076.9699707, 2773.1767578

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## BFS NS instance: NS_A2_B1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -390.3378906, 1578.7467041, -274.4427185, 1115.0172119, -1505.3551025, 1853.1893311
1: -627.9822998, 1747.9838867, -442.2922974, 1233.8166504, -1861.7988281, 2190.2761230
2: -468.9877930, 2011.9674072, -329.9755554, 1421.4028320, -1890.3906250, 2341.9428711
3: -1008.8870239, 1801.4213867, -709.9766235, 1269.6837158, -2278.5708008, 2511.3974609
4: -808.4592896, 1883.4610596, -568.2381592, 1329.5943604, -2138.0532227, 2451.6989746

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7242379, upper bound: 1781.7266354
time: 0.54 seconds

## Relational analysis of NS_A2_B1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7241928, upper bound: 1781.7264707
time: 0.52 seconds

## BFS NS instance: NS_A2_B1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -388.5496521, 1572.0577393, -274.4427185, 1115.0172119, -1503.5668945, 1846.5003662
1: -625.2743530, 1740.4183350, -442.2922974, 1233.8166504, -1859.0909424, 2182.7106934
2: -466.9485779, 2003.2802734, -329.9755554, 1421.4028320, -1888.3513184, 2333.2558594
3: -1004.1314087, 1793.6044922, -709.9766235, 1269.6837158, -2273.8151855, 2503.5808105
4: -804.7737427, 1875.3979492, -568.2381592, 1329.5943604, -2134.3679199, 2443.6359863

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 28

## BFS NS instance: NS_A2_B1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -390.3378906, 1578.7467041, -279.3706665, 1134.2976074, -1524.6354980, 1858.1174316
1: -627.9822998, 1747.9838867, -450.4078979, 1255.0859375, -1883.0682373, 2198.3918457
2: -468.9877930, 2011.9674072, -336.0948792, 1446.0902100, -1915.0780029, 2348.0622559
3: -1008.8870239, 1801.4213867, -722.8035278, 1291.5921631, -2300.4790039, 2524.2248535
4: -808.4592896, 1883.4610596, -578.5453491, 1352.6071777, -2161.0664062, 2462.0058594

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 28

## BFS NS instance: NS_A2_B1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -388.5496521, 1572.0577393, -279.3706665, 1134.2976074, -1522.8472900, 1851.4284668
1: -625.2743530, 1740.4183350, -450.4078979, 1255.0859375, -1880.3603516, 2190.8261719
2: -466.9485779, 2003.2802734, -336.0948792, 1446.0902100, -1913.0385742, 2339.3752441
3: -1004.1314087, 1793.6044922, -722.8035278, 1291.5921631, -2295.7236328, 2516.4079590
4: -804.7737427, 1875.3979492, -578.5453491, 1352.6071777, -2157.3808594, 2453.9428711

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 28

## BFS NS instance: NS_A2_B1_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -482.2149048, 1934.4200439, -274.4427185, 1115.0172119, -1597.2321777, 2208.8625488
1: -779.1491699, 2147.2224121, -442.2922974, 1233.8166504, -2012.9658203, 2589.5146484
2: -582.5481567, 2468.1884766, -329.9755554, 1421.4028320, -2003.9509277, 2798.1635742
3: -1254.6140137, 2215.2812500, -709.9766235, 1269.6837158, -2524.2976074, 2925.2573242
4: -1003.0828857, 2313.4118652, -568.2381592, 1329.5943604, -2332.6767578, 2881.6499023

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B1_A2_A1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7158082, upper bound: 1781.7190473
time: 0.80 seconds

## Relational analysis of NS_A2_B1_B1_A2_A1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7173608, upper bound: 1781.7207544
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -482.2149048, 1934.4200439, -279.3706665, 1134.2976074, -1616.5124512, 2213.7907715
1: -779.1491699, 2147.2224121, -450.4078979, 1255.0859375, -2034.2351074, 2597.6303711
2: -582.5481567, 2468.1884766, -336.0948792, 1446.0902100, -2028.6383057, 2804.2832031
3: -1254.6140137, 2215.2812500, -722.8035278, 1291.5921631, -2546.2060547, 2938.0847168
4: -1003.0828857, 2313.4118652, -578.5453491, 1352.6071777, -2355.6899414, 2891.9567871

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B1_A2_A1_B2_A1

### Relational analysis result of NS_A2_B1_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7158082, upper bound: 1781.7190473
time: 0.54 seconds

## Relational analysis of NS_A2_B1_B1_A2_A1_B2_A2

### Relational analysis result of NS_A2_B1_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7173608, upper bound: 1781.7207544
time: 0.49 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -481.1869202, 1930.1661377, -274.4427185, 1115.0172119, -1596.2041016, 2204.6083984
1: -777.5044556, 2142.6093750, -442.2922974, 1233.8166504, -2011.3210449, 2584.9016113
2: -581.3374023, 2462.7590332, -329.9755554, 1421.4028320, -2002.7402344, 2792.7341309
3: -1252.0045166, 2210.4218750, -709.9766235, 1269.6837158, -2521.6882324, 2920.3977051
4: -1000.9742432, 2308.2561035, -568.2381592, 1329.5943604, -2330.5683594, 2876.4941406

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B1_A2_A2_B1_A1

### Relational analysis result of NS_A2_B1_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7139555, upper bound: 1781.7169148
time: 0.59 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7147153, upper bound: 1781.7184688
time: 0.56 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -481.1869202, 1930.1661377, -279.3706665, 1134.2976074, -1615.4844971, 2209.5366211
1: -777.5044556, 2142.6093750, -450.4078979, 1255.0859375, -2032.5903320, 2593.0166016
2: -581.3374023, 2462.7590332, -336.0948792, 1446.0902100, -2027.4276123, 2798.8537598
3: -1252.0045166, 2210.4218750, -722.8035278, 1291.5921631, -2543.5966797, 2933.2253418
4: -1000.9742432, 2308.2561035, -578.5453491, 1352.6071777, -2353.5815430, 2886.8012695

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B1_A2_A2_B2_A1

### Relational analysis result of NS_A2_B1_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7139555, upper bound: 1781.7169148
time: 0.54 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2_B2_A2

### Relational analysis result of NS_A2_B1_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7147153, upper bound: 1781.7184688
time: 0.54 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -388.1438904, 1570.0700684, -369.6397400, 1482.5614014, -1870.7052002, 1939.7097168
1: -624.4998779, 1738.3515625, -598.4256592, 1646.1999512, -2270.6997070, 2336.7773438
2: -466.3586121, 2000.9025879, -447.2619019, 1892.4997559, -2358.8583984, 2448.1643066
3: -1003.3052979, 1791.3928223, -964.0110474, 1698.5567627, -2701.8620605, 2754.4746094
4: -803.9530029, 1872.9388428, -769.6611938, 1774.5987549, -2578.5515137, 2642.5998535

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 36

## BFS NS instance: NS_A2_B1_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -386.6103516, 1564.3378906, -369.6397400, 1482.5614014, -1869.1717529, 1933.9774170
1: -622.1891479, 1731.8579102, -598.4256592, 1646.1999512, -2268.3891602, 2330.2836914
2: -464.6183167, 1993.4416504, -447.2619019, 1892.4997559, -2357.1181641, 2440.7036133
3: -999.1990967, 1784.6806641, -964.0110474, 1698.5567627, -2697.7558594, 2747.9895020
4: -800.7770386, 1866.0429688, -769.6611938, 1774.5987549, -2575.3752441, 2635.7041016

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 36

## BFS NS instance: NS_A2_B1_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -482.2149048, 1934.4200439, -369.6397400, 1482.5614014, -1964.7763672, 2304.0595703
1: -779.1491699, 2147.2224121, -598.4256592, 1646.1999512, -2425.3491211, 2745.6479492
2: -582.5481567, 2468.1884766, -447.2619019, 1892.4997559, -2475.0478516, 2915.4504395
3: -1254.6140137, 2215.2812500, -964.0110474, 1698.5567627, -2952.2219238, 3176.9301758
4: -1003.0828857, 2313.4118652, -769.6611938, 1774.5987549, -2777.6806641, 3083.0729980

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B2_B1_A2_A1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7107335, upper bound: 1781.7089515
time: 0.53 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_A1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7111626, upper bound: 1781.7094599
time: 0.51 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -481.1869202, 1930.1661377, -369.6397400, 1482.5614014, -1963.7482910, 2299.8054199
1: -777.5044556, 2142.6093750, -598.4256592, 1646.1999512, -2423.7043457, 2741.0346680
2: -581.3374023, 2462.7590332, -447.2619019, 1892.4997559, -2473.8371582, 2910.0209961
3: -1252.0045166, 2210.4218750, -964.0110474, 1698.5567627, -2949.6843262, 3172.2661133
4: -1000.9742432, 2308.2561035, -769.6611938, 1774.5987549, -2775.5727539, 3077.9172363

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B2_B1_A2_A2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7107335, upper bound: 1781.7089515
time: 0.68 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_A2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7111626, upper bound: 1781.7094599
time: 0.49 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -388.1438904, 1570.0700684, -369.2091370, 1480.5931396, -1868.7369385, 1939.2791748
1: -624.4998779, 1738.3515625, -597.6428223, 1644.0800781, -2268.5798340, 2335.9941406
2: -466.3586121, 2000.9025879, -446.7429504, 1890.0368652, -2356.3955078, 2447.6455078
3: -1003.3052979, 1791.3928223, -962.7859497, 1696.2437744, -2699.5490723, 2753.3059082
4: -803.9530029, 1872.9388428, -768.7138062, 1772.2023926, -2576.1552734, 2641.6525879

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 36

## BFS NS instance: NS_A2_B1_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -386.6103516, 1564.3378906, -369.2091370, 1480.5931396, -1867.2034912, 1933.5469971
1: -622.1891479, 1731.8579102, -597.6428223, 1644.0800781, -2266.2692871, 2329.5004883
2: -464.6183167, 1993.4416504, -446.7429504, 1890.0368652, -2354.6552734, 2440.1845703
3: -999.1990967, 1784.6806641, -962.7859497, 1696.2437744, -2695.4428711, 2746.8208008
4: -800.7770386, 1866.0429688, -768.7138062, 1772.2023926, -2572.9794922, 2634.7565918

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 36

## BFS NS instance: NS_A2_B1_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -482.2149048, 1934.4200439, -369.2091370, 1480.5931396, -1962.8081055, 2303.6289062
1: -779.1491699, 2147.2224121, -597.6428223, 1644.0800781, -2423.2292480, 2744.8649902
2: -582.5481567, 2468.1884766, -446.7429504, 1890.0368652, -2472.5849609, 2914.9313965
3: -1254.6140137, 2215.2812500, -962.7859497, 1696.2437744, -2950.1059570, 3175.7614746
4: -1003.0828857, 2313.4118652, -768.7138062, 1772.2023926, -2775.2851562, 3082.1254883

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B2_B2_A2_A1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7107335, upper bound: 1781.7106394
time: 0.53 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_A1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7111626, upper bound: 1781.7111475
time: 0.51 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -481.1869202, 1930.1661377, -369.2091370, 1480.5931396, -1961.7800293, 2299.3747559
1: -777.5044556, 2142.6093750, -597.6428223, 1644.0800781, -2421.5844727, 2740.2512207
2: -581.3374023, 2462.7590332, -446.7429504, 1890.0368652, -2471.3742676, 2909.5019531
3: -1252.0045166, 2210.4218750, -962.7859497, 1696.2437744, -2947.5683594, 3171.0974121
4: -1000.9742432, 2308.2561035, -768.7138062, 1772.2023926, -2773.1767578, 3076.9699707

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B2_B2_A2_A2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7107335, upper bound: 1781.7103939
time: 0.64 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_A2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7111626, upper bound: 1781.7108926
time: 0.52 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -366.0052185, 1479.9211426, -366.0052185, 1479.9211426, -1845.9260254, 1845.9260254
1: -589.1425781, 1638.8649902, -589.1425781, 1638.8649902, -2228.0075684, 2228.0075684
2: -440.0267944, 1886.3773193, -440.0267944, 1886.3773193, -2326.4040527, 2326.4040527
3: -946.3379517, 1689.0131836, -946.3379517, 1689.0131836, -2635.3510742, 2635.3510742
4: -758.7982178, 1765.4758301, -758.7982178, 1765.4758301, -2524.2739258, 2524.2739258

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7341768, upper bound: 1781.7347160
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7342633, upper bound: 1781.7342628
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -366.0052185, 1479.9211426, -417.2734680, 1687.4654541, -2053.4704590, 1897.1944580
1: -589.1425781, 1638.8649902, -670.7286987, 1869.1072998, -2458.2492676, 2309.5937500
2: -440.0267944, 1886.3773193, -501.1014709, 2150.9934082, -2591.0202637, 2387.4787598
3: -946.3379517, 1689.0131836, -1078.9874268, 1924.5645752, -2870.9025879, 2768.0004883
4: -758.7982178, 1765.4758301, -864.3917847, 2012.0499268, -2770.8479004, 2629.8671875

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7341768, upper bound: 1781.7349482
time: 0.53 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7342633, upper bound: 1781.7345545
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -417.2734680, 1687.4654541, -366.0052185, 1479.9211426, -1897.1944580, 2053.4704590
1: -670.7286987, 1869.1072998, -589.1425781, 1638.8649902, -2309.5937500, 2458.2492676
2: -501.1014709, 2150.9934082, -440.0267944, 1886.3773193, -2387.4785156, 2591.0202637
3: -1078.9874268, 1924.5645752, -946.3379517, 1689.0131836, -2768.0004883, 2870.9025879
4: -864.3917847, 2012.0499268, -758.7982178, 1765.4758301, -2629.8671875, 2770.8479004

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7346655, upper bound: 1781.7341761
time: 0.50 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7342648, upper bound: 1781.7342612
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -417.2734680, 1687.4654541, -417.2734680, 1687.4654541, -2104.7387695, 2104.7390137
1: -670.7286987, 1869.1072998, -670.7286987, 1869.1072998, -2539.8352051, 2539.8352051
2: -501.1014709, 2150.9934082, -501.1014709, 2150.9934082, -2652.0944824, 2652.0942383
3: -1078.9874268, 1924.5645752, -1078.9874268, 1924.5645752, -3003.5520020, 3003.5520020
4: -864.3917847, 2012.0499268, -864.3917847, 2012.0499268, -2876.4411621, 2876.4411621

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7341774, upper bound: 1781.7349967
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7342648, upper bound: 1781.7346113
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -366.0052185, 1479.9211426, -460.4322510, 1844.2117920, -2210.2170410, 1940.3532715
1: -589.1425781, 1638.8649902, -744.2117920, 2047.9725342, -2637.1152344, 2383.0766602
2: -440.0267944, 1886.3773193, -556.5748291, 2353.6630859, -2793.6899414, 2442.9519043
3: -946.3379517, 1689.0131836, -1198.3090820, 2113.7905273, -3060.1284180, 2885.9873047
4: -758.7982178, 1765.4758301, -958.5090332, 2206.7712402, -2965.5690918, 2723.9841309

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7041902, upper bound: 1781.7034770
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7111601, upper bound: 1781.7058771
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -417.2734680, 1687.4654541, -460.4322510, 1844.2117920, -2261.4853516, 2147.8977051
1: -670.7286987, 1869.1072998, -744.2117920, 2047.9725342, -2718.7011719, 2613.3186035
2: -501.1014709, 2150.9934082, -556.5748291, 2353.6630859, -2854.7644043, 2707.5676270
3: -1078.9874268, 1924.5645752, -1198.3090820, 2113.7905273, -3192.7778320, 3120.7109375
4: -864.3917847, 2012.0499268, -958.5090332, 2206.7712402, -3071.1625977, 2970.5581055

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_B1_A2_A1

### Relational analysis result of NS_A2_B2_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7124570, upper bound: 1781.7073621
time: 0.50 seconds

## Relational analysis of NS_A2_B2_A1_B2_B1_A2_A2

### Relational analysis result of NS_A2_B2_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7111602, upper bound: 1781.7058771
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -366.0052185, 1479.9211426, -483.0009155, 1938.7708740, -2304.7758789, 1962.9219971
1: -589.1425781, 1638.8649902, -779.0424194, 2151.9399414, -2741.0825195, 2417.9074707
2: -440.0267944, 1886.3773193, -582.8635864, 2473.3574219, -2913.3842773, 2469.2409668
3: -946.3379517, 1689.0131836, -1254.5689697, 2220.6840820, -3167.0219727, 2943.3298340
4: -758.7982178, 1765.4758301, -1004.4208984, 2317.7651367, -3076.5634766, 2769.8967285

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7041902, upper bound: 1781.7071787
time: 0.55 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7111602, upper bound: 1781.7075750
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -417.2734680, 1687.4654541, -483.0009155, 1938.7708740, -2356.0441895, 2170.4663086
1: -670.7286987, 1869.1072998, -779.0424194, 2151.9399414, -2822.6684570, 2648.1496582
2: -501.1014709, 2150.9934082, -582.8635864, 2473.3574219, -2974.4582520, 2733.8564453
3: -1078.9874268, 1924.5645752, -1254.5689697, 2220.6840820, -3299.6713867, 3178.4128418
4: -864.3917847, 2012.0499268, -1004.4208984, 2317.7651367, -3182.1567383, 3016.4707031

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7041902, upper bound: 1781.7043640
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7111602, upper bound: 1781.7082764
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -460.4322510, 1844.2117920, -364.9852600, 1475.9201660, -1936.3522949, 2209.1970215
1: -744.2117920, 2047.9725342, -587.5151367, 1634.4132080, -2378.6250000, 2635.4877930
2: -556.5748291, 2353.6630859, -438.7923889, 1881.2736816, -2437.8481445, 2792.4555664
3: -1198.3090820, 2113.7905273, -943.7299805, 1684.3536377, -2881.3315430, 3057.5205078
4: -958.5090332, 2206.7712402, -756.6802979, 1760.6086426, -2719.1171875, 2963.4506836

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_A1_B1_B1_A1

### Relational analysis result of NS_A2_B2_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7168110, upper bound: 1781.7176720
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A2_A1_B1_B1_A2

### Relational analysis result of NS_A2_B2_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7208891, upper bound: 1781.7206201
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -460.4322510, 1844.2117920, -460.4322510, 1844.2117920, -2304.6440430, 2304.6440430
1: -744.2117920, 2047.9725342, -744.2117920, 2047.9725342, -2792.1843262, 2792.1843262
2: -556.5748291, 2353.6630859, -556.5748291, 2353.6630859, -2910.2375488, 2910.2375488
3: -1198.3090820, 2113.7905273, -1198.3090820, 2113.7905273, -3309.2290039, 3309.2290039
4: -958.5090332, 2206.7712402, -958.5090332, 2206.7712402, -3165.2790527, 3165.2790527

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_A1_B1_B2_B1

### Relational analysis result of NS_A2_B2_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7051490, upper bound: 1781.7059136
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A2_A1_B1_B2_B2

### Relational analysis result of NS_A2_B2_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7051064, upper bound: 1781.7051064
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -460.4322510, 1844.2117920, -414.9363403, 1678.4685059, -2138.9008789, 2259.1481934
1: -744.2117920, 2047.9725342, -667.1145630, 1859.0468750, -2603.2587891, 2715.0871582
2: -556.5748291, 2353.6630859, -498.3149414, 2139.5361328, -2696.1103516, 2851.9780273
3: -1198.3090820, 2113.7905273, -1073.1156006, 1914.0841064, -3110.2390137, 3186.9062500
4: -958.5090332, 2206.7712402, -859.6168213, 2001.1215820, -2959.6296387, 3066.3876953

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_A1_B2_B1_B1

### Relational analysis result of NS_A2_B2_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7048083, upper bound: 1781.7047341
time: 0.48 seconds

## Relational analysis of NS_A2_B2_A2_A1_B2_B1_B2

### Relational analysis result of NS_A2_B2_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7049642, upper bound: 1781.7051994
time: 0.52 seconds

## BFS NS instance: NS_A2_B2_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -460.4322510, 1844.2117920, -482.8201904, 1938.0437012, -2398.4758301, 2327.0319824
1: -744.2117920, 2047.9725342, -778.7558594, 2151.1306152, -2895.3422852, 2826.7285156
2: -556.5748291, 2353.6630859, -582.6433105, 2472.4279785, -3029.0021973, 2936.3063965
3: -1198.3090820, 2113.7905273, -1254.0969238, 2219.8618164, -3414.9221191, 3366.1013184
4: -958.5090332, 2206.7712402, -1004.0454102, 2316.9011230, -3275.4094238, 3210.8156738

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_A1_B2_B2_B1

### Relational analysis result of NS_A2_B2_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7048083, upper bound: 1781.7047341
time: 0.49 seconds

## Relational analysis of NS_A2_B2_A2_A1_B2_B2_B2

### Relational analysis result of NS_A2_B2_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7049642, upper bound: 1781.7051994
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -480.3002014, 1927.8143311, -361.4742737, 1461.1080322, -1941.4082031, 2289.2885742
1: -774.6972046, 2139.8459473, -581.7064819, 1618.6309814, -2393.3281250, 2721.5524902
2: -579.6039429, 2459.3994141, -434.7811279, 1862.5925293, -2442.1965332, 2894.1801758
3: -1247.5859375, 2208.1982422, -934.8912964, 1667.9838867, -2915.0095215, 3143.0895996
4: -998.7920532, 2304.6918945, -750.0761108, 1742.6854248, -2741.4775391, 3054.7678223

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7036141, upper bound: 1781.7037621
time: 0.51 seconds

## Relational analysis of NS_A2_B2_A2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7036141, upper bound: 1781.7037621
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -478.7261353, 1921.3687744, -361.0342407, 1459.2850342, -1938.0111084, 2282.4023438
1: -772.1367798, 2132.7263184, -580.9943848, 1616.6287842, -2388.7653809, 2713.7207031
2: -577.7418213, 2451.1293945, -434.2512817, 1860.2604980, -2438.0024414, 2885.3803711
3: -1243.5155029, 2200.9172363, -933.7692871, 1665.9406738, -2908.9755859, 3134.6865234
4: -995.5452271, 2296.9990234, -749.1805420, 1740.5278320, -2736.0727539, 3046.1796875

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7037854, upper bound: 1781.7037854
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7037854, upper bound: 1781.7037854
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -483.0009155, 1938.7708740, -415.8037415, 1681.8759766, -2164.8769531, 2354.5742188
1: -779.0424194, 2151.9399414, -668.4752197, 1862.8389893, -2641.8813477, 2820.4150391
2: -582.8635864, 2473.3574219, -499.3606262, 2143.8862305, -2726.7497559, 2972.7175293
3: -1254.5689697, 2220.6840820, -1075.3530273, 1917.9638672, -3171.8183594, 3296.0368652
4: -1004.4208984, 2317.7651367, -861.4062500, 2005.1766357, -3009.5976562, 3179.1713867

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7044731, upper bound: 1781.7047592
time: 0.53 seconds

## Relational analysis of NS_A2_B2_A2_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7049799, upper bound: 1781.7049799
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -483.0009155, 1938.7708740, -483.0009155, 1938.7708740, -2421.7717285, 2421.7717285
1: -779.0424194, 2151.9399414, -779.0424194, 2151.9399414, -2930.9824219, 2930.9824219
2: -582.8635864, 2473.3574219, -582.8635864, 2473.3574219, -3056.2209473, 3056.2209473
3: -1254.5689697, 2220.6840820, -1254.5689697, 2220.6840820, -3473.4011230, 3473.4011230
4: -1004.4208984, 2317.7651367, -1004.4208984, 2317.7651367, -3322.1860352, 3322.1860352

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7044731, upper bound: 1781.7047592
time: 0.55 seconds

## Relational analysis of NS_A2_B2_A2_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7049799, upper bound: 1781.7049799
time: 0.50 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.04 seconds
NS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7347025, upper bound: 1781.7324269
NS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7307421, upper bound: 1781.7307421
NS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7347025, upper bound: 1781.7332929
NS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7307421, upper bound: 1781.7314321
NS_A1_B1_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7332929, upper bound: 1781.7346340
NS_A1_B1_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7314321, upper bound: 1781.7310731
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7340968, upper bound: 1781.7331355
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7314321, upper bound: 1781.7322615
NS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7266354, upper bound: 1781.7242379
NS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7264707, upper bound: 1781.7241928
NS_A1_B2_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7190473, upper bound: 1781.7158082
NS_A1_B2_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7207544, upper bound: 1781.7173608
NS_A1_B2_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7190473, upper bound: 1781.7158082
NS_A1_B2_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7207544, upper bound: 1781.7173608
NS_A1_B2_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7169148, upper bound: 1781.7139555
NS_A1_B2_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7184688, upper bound: 1781.7147153
NS_A1_B2_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7169148, upper bound: 1781.7139581
NS_A1_B2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7184688, upper bound: 1781.7147153
NS_A2_B1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7242379, upper bound: 1781.7266354
NS_A2_B1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7241928, upper bound: 1781.7264707
NS_A2_B1_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7158082, upper bound: 1781.7190473
NS_A2_B1_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7173608, upper bound: 1781.7207544
NS_A2_B1_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7158082, upper bound: 1781.7190473
NS_A2_B1_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7173608, upper bound: 1781.7207544
NS_A2_B1_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7139555, upper bound: 1781.7169148
NS_A2_B1_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7147153, upper bound: 1781.7184688
NS_A2_B1_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7139555, upper bound: 1781.7169148
NS_A2_B1_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7147153, upper bound: 1781.7184688
NS_A2_B1_B2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7107335, upper bound: 1781.7089515
NS_A2_B1_B2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7111626, upper bound: 1781.7094599
NS_A2_B1_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7107335, upper bound: 1781.7089515
NS_A2_B1_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7111626, upper bound: 1781.7094599
NS_A2_B1_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7107335, upper bound: 1781.7106394
NS_A2_B1_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7111626, upper bound: 1781.7111475
NS_A2_B1_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7107335, upper bound: 1781.7103939
NS_A2_B1_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7111626, upper bound: 1781.7108926
NS_A2_B2_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7341768, upper bound: 1781.7347160
NS_A2_B2_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7342633, upper bound: 1781.7342628
NS_A2_B2_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7341768, upper bound: 1781.7349482
NS_A2_B2_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7342633, upper bound: 1781.7345545
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7346655, upper bound: 1781.7341761
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7342648, upper bound: 1781.7342612
NS_A2_B2_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7341774, upper bound: 1781.7349967
NS_A2_B2_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7342648, upper bound: 1781.7346113
NS_A2_B2_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7041902, upper bound: 1781.7034770
NS_A2_B2_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7111601, upper bound: 1781.7058771
NS_A2_B2_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7124570, upper bound: 1781.7073621
NS_A2_B2_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7111602, upper bound: 1781.7058771
NS_A2_B2_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7041902, upper bound: 1781.7071787
NS_A2_B2_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7111602, upper bound: 1781.7075750
NS_A2_B2_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7041902, upper bound: 1781.7043640
NS_A2_B2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7111602, upper bound: 1781.7082764
NS_A2_B2_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7168110, upper bound: 1781.7176720
NS_A2_B2_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7208891, upper bound: 1781.7206201
NS_A2_B2_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7051490, upper bound: 1781.7059136
NS_A2_B2_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7051064, upper bound: 1781.7051064
NS_A2_B2_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7048083, upper bound: 1781.7047341
NS_A2_B2_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7049642, upper bound: 1781.7051994
NS_A2_B2_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7048083, upper bound: 1781.7047341
NS_A2_B2_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7049642, upper bound: 1781.7051994
NS_A2_B2_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7036141, upper bound: 1781.7037621
NS_A2_B2_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7036141, upper bound: 1781.7037621
NS_A2_B2_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7037854, upper bound: 1781.7037854
NS_A2_B2_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7037854, upper bound: 1781.7037854
NS_A2_B2_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7044731, upper bound: 1781.7047592
NS_A2_B2_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7049799, upper bound: 1781.7049799
NS_A2_B2_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7044731, upper bound: 1781.7047592
NS_A2_B2_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.04
Output dim: 0, lower bound: -1781.7049799, upper bound: 1781.7049799

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -270.9939575, 1100.6661377, -273.1428833, 1109.6066895, -1380.6002197, 1373.8089600
1: -436.7276306, 1218.1689453, -440.1948242, 1227.9208984, -1664.6485596, 1658.3636475
2: -325.8062134, 1403.1060791, -328.4038696, 1414.5039062, -1740.3100586, 1731.5098877
3: -701.1478882, 1253.3719482, -706.6502686, 1263.5389404, -1964.6867676, 1960.0220947
4: -560.9750977, 1312.4527588, -565.5025635, 1323.1314697, -1884.1063232, 1877.9552002

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7307421, upper bound: 1781.7307421
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7307421, upper bound: 1781.7307421
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -270.5726624, 1099.1437988, -272.1133118, 1105.6427002, -1376.2153320, 1371.2570801
1: -436.3395691, 1216.2727051, -438.4892883, 1223.5072021, -1659.8465576, 1654.7618408
2: -325.4986572, 1400.9833984, -327.2037659, 1409.4437256, -1734.9423828, 1728.1870117
3: -699.9791870, 1251.5307617, -703.9307861, 1258.9621582, -1958.9414062, 1955.4615479
4: -560.2523804, 1310.4941406, -563.4880981, 1318.2532959, -1878.5054932, 1873.9821777

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7307421, upper bound: 1781.7307421
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7307421, upper bound: 1781.7307421
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -270.9939575, 1100.6661377, -278.1034241, 1129.0369873, -1400.0310059, 1378.7694092
1: -436.7276306, 1218.1689453, -448.3612366, 1249.3507080, -1686.0783691, 1666.5301514
2: -325.8062134, 1403.1060791, -334.5608215, 1439.3833008, -1765.1894531, 1737.6667480
3: -701.1478882, 1253.3719482, -719.5600586, 1285.5980225, -1986.7458496, 1972.9318848
4: -560.9750977, 1312.4527588, -575.8739014, 1346.3167725, -1907.2916260, 1888.3264160

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7310731, upper bound: 1781.7314321
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7310731, upper bound: 1781.7314321
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -270.5726624, 1099.1437988, -276.9013367, 1124.3590088, -1394.9316406, 1376.0451660
1: -436.3395691, 1216.2727051, -446.3817749, 1244.1622314, -1680.5018311, 1662.6545410
2: -325.4986572, 1400.9833984, -333.1553040, 1433.4006348, -1758.8992920, 1734.1386719
3: -699.9791870, 1251.5307617, -716.3927612, 1280.2344971, -1980.2136230, 1967.9235840
4: -560.2523804, 1310.4941406, -573.5050659, 1340.5820312, -1900.8343506, 1883.9992676

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7310731, upper bound: 1781.7314321
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7310731, upper bound: 1781.7314321
time: 0.49 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -278.1034241, 1129.0369873, -270.9939575, 1100.6661377, -1378.7694092, 1400.0308838
1: -448.3612366, 1249.3507080, -436.7276306, 1218.1689453, -1666.5301514, 1686.0783691
2: -334.5608215, 1439.3833008, -325.8062134, 1403.1060791, -1737.6666260, 1765.1894531
3: -719.5600586, 1285.5980225, -701.1478882, 1253.3719482, -1972.9318848, 1986.7458496
4: -575.8739014, 1346.3167725, -560.9750977, 1312.4527588, -1888.3265381, 1907.2916260

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7314321, upper bound: 1781.7310731
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7314321, upper bound: 1781.7310731
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -276.9013367, 1124.3590088, -270.5726624, 1099.1437988, -1376.0451660, 1394.9316406
1: -446.3817749, 1244.1622314, -436.3395691, 1216.2727051, -1662.6545410, 1680.5018311
2: -333.1553040, 1433.4006348, -325.4986572, 1400.9833984, -1734.1386719, 1758.8992920
3: -716.3927612, 1280.2344971, -699.9791870, 1251.5307617, -1967.9235840, 1980.2136230
4: -573.5050659, 1340.5820312, -560.2523804, 1310.4941406, -1883.9992676, 1900.8342285

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7314321, upper bound: 1781.7310731
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7314321, upper bound: 1781.7310731
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -276.0092163, 1120.3508301, -278.1034241, 1129.0369873, -1405.0461426, 1398.4539795
1: -444.9826965, 1239.8773193, -448.3612366, 1249.3507080, -1694.3333740, 1688.2385254
2: -332.0290833, 1428.3106689, -334.5608215, 1439.3833008, -1771.4123535, 1762.8713379
3: -714.2047119, 1275.6909180, -719.5600586, 1285.5980225, -1999.8027344, 1995.2509766
4: -571.4569092, 1335.9312744, -575.8739014, 1346.3167725, -1917.7736816, 1911.8049316

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7322388, upper bound: 1781.7322615
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7322388, upper bound: 1781.7322615
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -276.2135315, 1121.1513672, -276.9013367, 1124.3590088, -1400.5723877, 1398.0527344
1: -445.5941467, 1240.6939697, -446.3817749, 1244.1622314, -1689.7563477, 1687.0756836
2: -332.5011292, 1429.2166748, -333.1553040, 1433.4006348, -1765.9017334, 1762.3719482
3: -714.9195557, 1276.7307129, -716.3927612, 1280.2344971, -1995.1538086, 1993.1235352
4: -572.0524292, 1336.9727783, -573.5050659, 1340.5820312, -1912.6343994, 1910.4777832

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7322388, upper bound: 1781.7322615
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7322388, upper bound: 1781.7322615
time: 0.51 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -270.9939575, 1100.6661377, -388.8701477, 1572.6916504, -1843.6855469, 1489.5362549
1: -436.7276306, 1218.1689453, -625.6098022, 1741.3771973, -2178.1047363, 1843.7788086
2: -325.8062134, 1403.1060791, -467.2184143, 2004.2473145, -2330.0532227, 1870.3243408
3: -701.1478882, 1253.3719482, -1005.1193848, 1794.5133057, -2495.6604004, 2258.4912109
4: -560.9750977, 1312.4527588, -805.3878174, 1876.2062988, -2437.1806641, 2117.8403320

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7299969, upper bound: 1781.7294467
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7299969, upper bound: 1781.7294467
time: 0.54 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -270.5726624, 1099.1437988, -387.9901123, 1569.3405762, -1839.9132080, 1487.1339111
1: -436.3395691, 1216.2727051, -624.1649170, 1737.6225586, -2173.9619141, 1840.4376221
2: -325.4986572, 1400.9833984, -466.1880188, 1999.9447021, -2325.4428711, 1867.1713867
3: -699.9791870, 1251.5307617, -1002.8140869, 1790.6657715, -2490.6450195, 2254.3447266
4: -560.2523804, 1310.4941406, -803.6473389, 1872.0844727, -2432.3361816, 2114.1416016

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7299969, upper bound: 1781.7294467
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7299969, upper bound: 1781.7294467
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -269.7445374, 1096.1376953, -477.8624268, 1916.8535156, -2186.5981445, 1573.9996338
1: -434.5365601, 1212.8875732, -772.1135254, 2127.2194824, -2561.7561035, 1985.0009766
2: -324.2350769, 1397.3234863, -577.3088989, 2446.2294922, -2770.4645996, 1974.6323242
3: -697.6375122, 1247.8240967, -1243.1137695, 2193.4692383, -2891.1066895, 2490.8986816
4: -558.3994751, 1306.8200684, -993.6220703, 2291.7473145, -2850.1467285, 2300.4421387

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -273.2131958, 1110.0550537, -479.7280884, 1924.1412354, -2197.3540039, 1589.7832031
1: -440.3440857, 1228.3277588, -775.1840210, 2135.9216309, -2576.2656250, 2003.5117188
2: -328.5141602, 1415.1093750, -579.5617065, 2455.1589355, -2783.6726074, 1994.6711426
3: -706.8873901, 1263.8894043, -1248.2899170, 2203.5061035, -2910.3933105, 2512.1791992
4: -565.6705933, 1323.6418457, -997.8940430, 2301.1623535, -2866.8327637, 2321.5358887

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -274.8045044, 1116.0301514, -477.8624268, 1916.8535156, -2191.6579590, 1593.8920898
1: -442.8558350, 1234.8051758, -772.1135254, 2127.2194824, -2570.0751953, 2006.9187012
2: -330.5003967, 1422.7670898, -577.3088989, 2446.2294922, -2776.7299805, 2000.0759277
3: -710.7507324, 1270.4899902, -1243.1137695, 2193.4692383, -2904.2199707, 2513.3750000
4: -568.9776001, 1330.5937500, -993.6220703, 2291.7473145, -2860.7248535, 2324.2155762

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -278.0391235, 1128.8770752, -479.7280884, 1924.1412354, -2202.1801758, 1608.6051025
1: -448.2961731, 1249.1035156, -775.1840210, 2135.9216309, -2584.2177734, 2024.2874756
2: -334.5107727, 1439.2188721, -579.5617065, 2455.1589355, -2789.6691895, 2018.7805176
3: -719.4544678, 1285.3074951, -1248.2899170, 2203.5061035, -2922.9604492, 2533.5705566
4: -575.7686157, 1346.1269531, -997.8940430, 2301.1623535, -2876.9309082, 2344.0207520

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -269.7445374, 1096.1376953, -477.3110352, 1914.4456787, -2184.1901855, 1573.4484863
1: -434.5365601, 1212.8875732, -771.2289429, 2124.6833496, -2559.2199707, 1984.1164551
2: -324.2350769, 1397.3234863, -576.6677856, 2443.1494141, -2767.3842773, 1973.9912109
3: -697.6375122, 1247.8240967, -1241.7421875, 2190.7302246, -2888.3676758, 2489.5661621
4: -558.3994751, 1306.8200684, -992.4887695, 2288.8159180, -2847.2153320, 2299.3088379

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -273.2131958, 1110.0550537, -478.6673584, 1919.7742920, -2192.9870605, 1588.7224121
1: -440.3440857, 1228.3277588, -773.4877319, 2131.1782227, -2571.5222168, 2001.8154297
2: -328.5141602, 1415.1093750, -578.3104858, 2449.5825195, -2778.0964355, 1993.4199219
3: -706.8873901, 1263.8894043, -1245.5916748, 2198.5061035, -2905.3933105, 2509.4809570
4: -565.6705933, 1323.6418457, -995.7145386, 2295.8601074, -2861.5307617, 2319.3564453

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -274.8045044, 1116.0301514, -477.3110352, 1914.4456787, -2189.2502441, 1593.3409424
1: -442.8558350, 1234.8051758, -771.2289429, 2124.6833496, -2567.5390625, 2006.0341797
2: -330.5003967, 1422.7670898, -576.6677856, 2443.1494141, -2773.6496582, 1999.4348145
3: -710.7507324, 1270.4899902, -1241.7421875, 2190.7302246, -2901.4807129, 2512.0759277
4: -568.9776001, 1330.5937500, -992.4887695, 2288.8159180, -2857.7934570, 2323.0822754

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -278.0391235, 1128.8770752, -478.6673584, 1919.7742920, -2197.8132324, 1607.5443115
1: -448.2961731, 1249.1035156, -773.4877319, 2131.1782227, -2579.4743652, 2022.5913086
2: -334.5107727, 1439.2188721, -578.3104858, 2449.5825195, -2784.0932617, 2017.5292969
3: -719.4544678, 1285.3074951, -1245.5916748, 2198.5061035, -2917.9604492, 2530.8991699
4: -575.7686157, 1346.1269531, -995.7145386, 2295.8601074, -2871.6286621, 2341.8410645

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -388.8701477, 1572.6916504, -270.9939575, 1100.6661377, -1489.5362549, 1843.6855469
1: -625.6098022, 1741.3771973, -436.7276306, 1218.1689453, -1843.7788086, 2178.1047363
2: -467.2184143, 2004.2473145, -325.8062134, 1403.1060791, -1870.3244629, 2330.0532227
3: -1005.1193848, 1794.5133057, -701.1478882, 1253.3719482, -2258.4912109, 2495.6604004
4: -805.3878174, 1876.2062988, -560.9750977, 1312.4527588, -2117.8403320, 2437.1806641

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7294467, upper bound: 1781.7299969
time: 0.54 seconds

## Relational analysis of NS_A2_B1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7294467, upper bound: 1781.7299969
time: 0.50 seconds

## BFS NS instance: NS_A2_B1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -387.9901123, 1569.3405762, -270.5726624, 1099.1437988, -1487.1339111, 1839.9132080
1: -624.1649170, 1737.6225586, -436.3395691, 1216.2727051, -1840.4376221, 2173.9619141
2: -466.1880188, 1999.9447021, -325.4986572, 1400.9833984, -1867.1713867, 2325.4428711
3: -1002.8140869, 1790.6657715, -699.9791870, 1251.5307617, -2254.3447266, 2490.6450195
4: -803.6473389, 1872.0844727, -560.2523804, 1310.4941406, -2114.1416016, 2432.3361816

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7294467, upper bound: 1781.7299969
time: 0.47 seconds

## Relational analysis of NS_A2_B1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7294467, upper bound: 1781.7299969
time: 0.51 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -477.8624268, 1916.8535156, -269.7445374, 1096.1376953, -1573.9996338, 2186.5981445
1: -772.1135254, 2127.2194824, -434.5365601, 1212.8875732, -1985.0010986, 2561.7561035
2: -577.3088989, 2446.2294922, -324.2350769, 1397.3234863, -1974.6323242, 2770.4645996
3: -1243.1137695, 2193.4692383, -697.6375122, 1247.8240967, -2490.8986816, 2891.1066895
4: -993.6220703, 2291.7473145, -558.3994751, 1306.8200684, -2300.4421387, 2850.1467285

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_B1_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -479.7280884, 1924.1412354, -273.2131958, 1110.0550537, -1589.7832031, 2197.3540039
1: -775.1840210, 2135.9216309, -440.3440857, 1228.3277588, -2003.5117188, 2576.2656250
2: -579.5617065, 2455.1589355, -328.5141602, 1415.1093750, -1994.6711426, 2783.6726074
3: -1248.2899170, 2203.5061035, -706.8873901, 1263.8894043, -2512.1791992, 2910.3933105
4: -997.8940430, 2301.1623535, -565.6705933, 1323.6418457, -2321.5358887, 2866.8330078

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B1_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_B1_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -477.8624268, 1916.8535156, -274.8045044, 1116.0301514, -1593.8920898, 2191.6579590
1: -772.1135254, 2127.2194824, -442.8558350, 1234.8051758, -2006.9187012, 2570.0751953
2: -577.3088989, 2446.2294922, -330.5003967, 1422.7670898, -2000.0759277, 2776.7299805
3: -1243.1137695, 2193.4692383, -710.7507324, 1270.4899902, -2513.3752441, 2904.2199707
4: -993.6220703, 2291.7473145, -568.9776001, 1330.5937500, -2324.2155762, 2860.7248535

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B1_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B1_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_B1_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -479.7280884, 1924.1412354, -278.0391235, 1128.8770752, -1608.6051025, 2202.1801758
1: -775.1840210, 2135.9216309, -448.2961731, 1249.1035156, -2024.2874756, 2584.2177734
2: -579.5617065, 2455.1589355, -334.5107727, 1439.2188721, -2018.7805176, 2789.6691895
3: -1248.2899170, 2203.5061035, -719.4544678, 1285.3074951, -2533.5705566, 2922.9604492
4: -997.8940430, 2301.1623535, -575.7686157, 1346.1269531, -2344.0207520, 2876.9306641

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B1_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B1_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_B1_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -477.3110352, 1914.4456787, -269.7445374, 1096.1376953, -1573.4484863, 2184.1901855
1: -771.2289429, 2124.6833496, -434.5365601, 1212.8875732, -1984.1164551, 2559.2199707
2: -576.6677856, 2443.1494141, -324.2350769, 1397.3234863, -1973.9912109, 2767.3842773
3: -1241.7421875, 2190.7302246, -697.6375122, 1247.8240967, -2489.5661621, 2888.3676758
4: -992.4887695, 2288.8159180, -558.3994751, 1306.8200684, -2299.3088379, 2847.2153320

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_B1_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -478.6673584, 1919.7742920, -273.2131958, 1110.0550537, -1588.7224121, 2192.9870605
1: -773.4877319, 2131.1782227, -440.3440857, 1228.3277588, -2001.8154297, 2571.5222168
2: -578.3104858, 2449.5825195, -328.5141602, 1415.1093750, -1993.4199219, 2778.0964355
3: -1245.5916748, 2198.5061035, -706.8873901, 1263.8894043, -2509.4809570, 2905.3933105
4: -995.7145386, 2295.8601074, -565.6705933, 1323.6418457, -2319.3564453, 2861.5307617

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_B1_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -477.3110352, 1914.4456787, -274.8045044, 1116.0301514, -1593.3409424, 2189.2502441
1: -771.2289429, 2124.6833496, -442.8558350, 1234.8051758, -2006.0341797, 2567.5390625
2: -576.6677856, 2443.1494141, -330.5003967, 1422.7670898, -1999.4348145, 2773.6496582
3: -1241.7421875, 2190.7302246, -710.7507324, 1270.4899902, -2512.0759277, 2901.4807129
4: -992.4887695, 2288.8159180, -568.9776001, 1330.5937500, -2323.0822754, 2857.7934570

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_B1_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -478.6673584, 1919.7742920, -278.0391235, 1128.8770752, -1607.5443115, 2197.8132324
1: -773.4877319, 2131.1782227, -448.2961731, 1249.1035156, -2022.5913086, 2579.4743652
2: -578.3104858, 2449.5825195, -334.5107727, 1439.2188721, -2017.5292969, 2784.0932617
3: -1245.5916748, 2198.5061035, -719.4544678, 1285.3074951, -2530.8991699, 2917.9604492
4: -995.7145386, 2295.8601074, -575.7686157, 1346.1269531, -2341.8413086, 2871.6286621

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B1_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B1_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_B2_B1_A2_A1_A1

### Backsubstitution after applying NS history:
0: -477.8624268, 1916.8535156, -363.3180237, 1457.0644531, -1934.9267578, 2280.1716309
1: -772.1135254, 2127.2194824, -588.2398071, 1617.9841309, -2390.0976562, 2715.4592285
2: -577.3088989, 2446.2294922, -439.6581421, 1860.0341797, -2437.3427734, 2885.8876953
3: -1243.1137695, 2193.4692383, -947.6914673, 1669.0634766, -2910.8298340, 3138.8544922
4: -993.6220703, 2291.7473145, -756.4496460, 1744.0117188, -2737.6333008, 3048.1970215

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B2_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B2_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A2_B1_B2_B1_A2_A1_A2

### Backsubstitution after applying NS history:
0: -479.7280884, 1924.1412354, -368.7907410, 1479.1027832, -1958.8308105, 2292.9318848
1: -775.1840210, 2135.9216309, -597.0691528, 1642.3903809, -2417.5744629, 2732.9907227
2: -579.5617065, 2455.1589355, -446.2419128, 1888.1129150, -2467.6743164, 2901.4006348
3: -1248.2899170, 2203.5061035, -961.8423462, 1694.5732422, -2941.7561035, 3162.8842773
4: -997.8940430, 2301.1623535, -767.8919067, 1770.4572754, -2768.3510742, 3069.0534668

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B2_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B2_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A2_B1_B2_B1_A2_A2_A1

### Backsubstitution after applying NS history:
0: -477.3110352, 1914.4456787, -363.3180237, 1457.0644531, -1934.3754883, 2277.7636719
1: -771.2289429, 2124.6833496, -588.2398071, 1617.9841309, -2389.2126465, 2712.9230957
2: -576.6677856, 2443.1494141, -439.6581421, 1860.0341797, -2436.7016602, 2882.8073730
3: -1241.7421875, 2190.7302246, -947.6914673, 1669.0634766, -2909.5307617, 3136.3017578
4: -992.4887695, 2288.8159180, -756.4496460, 1744.0117188, -2736.5004883, 3045.2656250

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B2_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B2_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B2_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A2_B1_B2_B1_A2_A2_A2

### Backsubstitution after applying NS history:
0: -478.6673584, 1919.7742920, -368.7907410, 1479.1027832, -1957.7701416, 2288.5649414
1: -773.4877319, 2131.1782227, -597.0691528, 1642.3903809, -2415.8781738, 2728.2470703
2: -578.3104858, 2449.5825195, -446.2419128, 1888.1129150, -2466.4233398, 2895.8244629
3: -1245.5916748, 2198.5061035, -961.8423462, 1694.5732422, -2939.1296387, 3158.0834961
4: -995.7145386, 2295.8601074, -767.8919067, 1770.4572754, -2766.1713867, 3063.7514648

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B2_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B2_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B2_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A2_B1_B2_B2_A2_A1_A1

### Backsubstitution after applying NS history:
0: -477.8624268, 1916.8535156, -362.4319763, 1453.3472900, -1931.2094727, 2279.2854004
1: -772.1135254, 2127.2194824, -586.7276611, 1613.9405518, -2386.0539551, 2713.9472656
2: -577.3088989, 2446.2294922, -438.5872498, 1855.3131104, -2432.6220703, 2884.8166504
3: -1243.1137695, 2193.4692383, -945.3026123, 1664.6783447, -2906.6437988, 3136.5471191
4: -993.6220703, 2291.7473145, -754.5301514, 1739.4521484, -2733.0739746, 3046.2773438

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B2_B2_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B2_B2_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_B2_B2_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B2_B2_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B2_B2_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A2_B1_B2_B2_A2_A1_A2

### Backsubstitution after applying NS history:
0: -479.7280884, 1924.1412354, -368.3590698, 1477.1511230, -1956.8791504, 2292.5002441
1: -775.1840210, 2135.9216309, -596.2831421, 1640.2799072, -2415.4638672, 2732.2048340
2: -579.5617065, 2455.1589355, -445.7213745, 1885.6690674, -2465.2302246, 2900.8801270
3: -1248.2899170, 2203.5061035, -960.6110229, 1692.2690430, -2939.6503906, 3161.7099609
4: -997.8940430, 2301.1623535, -766.9405518, 1768.0756836, -2765.9694824, 3068.1022949

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B2_B2_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B2_B2_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_B2_B2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B2_B2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B2_B2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A2_B1_B2_B2_A2_A2_A1

### Backsubstitution after applying NS history:
0: -477.3110352, 1914.4456787, -362.4319763, 1453.3472900, -1930.6582031, 2276.8771973
1: -771.2289429, 2124.6833496, -586.7276611, 1613.9405518, -2385.1689453, 2711.4111328
2: -576.6677856, 2443.1494141, -438.5872498, 1855.3131104, -2431.9809570, 2881.7365723
3: -1241.7421875, 2190.7302246, -945.3026123, 1664.6783447, -2905.3449707, 3133.9946289
4: -992.4887695, 2288.8159180, -754.5301514, 1739.4521484, -2731.9409180, 3043.3459473

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B2_B2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B2_B2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_B2_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B2_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B2_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A2_B1_B2_B2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -478.6673584, 1919.7742920, -368.3590698, 1477.1511230, -1955.8184814, 2288.1333008
1: -773.4877319, 2131.1782227, -596.2831421, 1640.2799072, -2413.7673340, 2727.4614258
2: -578.3104858, 2449.5825195, -445.7213745, 1885.6690674, -2463.9794922, 2895.3039551
3: -1245.5916748, 2198.5061035, -960.6110229, 1692.2690430, -2937.0239258, 3156.9094238
4: -995.7145386, 2295.8601074, -766.9405518, 1768.0756836, -2763.7897949, 3062.8002930

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B2_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B2_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_B2_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B2_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B2_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -357.1508179, 1444.1069336, -353.0356750, 1427.2999268, -1784.4506836, 1797.1425781
1: -575.0017700, 1599.2917480, -568.4414673, 1580.7686768, -2155.7705078, 2167.7331543
2: -429.3719788, 1840.7468262, -424.4314575, 1819.3449707, -2248.7167969, 2265.1782227
3: -924.0037231, 1647.5461426, -913.6499023, 1628.1541748, -2552.1569824, 2561.1960449
4: -740.1560059, 1722.6846924, -731.4971924, 1702.6254883, -2442.7805176, 2454.1818848

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7341673, upper bound: 1781.7341673
time: 0.53 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7341673, upper bound: 1781.7342097
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -360.3451538, 1456.6596680, -363.6524048, 1469.3791504, -1829.7242432, 1820.3120117
1: -579.9312744, 1613.1932373, -585.9936523, 1627.4024658, -2207.3337402, 2199.1865234
2: -433.1715393, 1856.8184814, -437.3565979, 1873.0939941, -2306.2656250, 2294.1748047
3: -931.8484497, 1662.3767090, -941.1972046, 1677.5747070, -2609.4230957, 2603.5739746
4: -747.1314697, 1737.8078613, -753.6337280, 1754.0329590, -2501.1645508, 2491.4416504

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7342094, upper bound: 1781.7341766
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7342094, upper bound: 1781.7342628
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -357.1508179, 1444.1069336, -402.8233032, 1628.8464355, -1985.9970703, 1846.9301758
1: -575.0017700, 1599.2917480, -647.6110229, 1804.3489990, -2379.3508301, 2246.9028320
2: -429.3719788, 1840.7468262, -483.7769470, 2076.2917480, -2505.6635742, 2324.5236816
3: -924.0037231, 1647.5461426, -1042.1881104, 1856.8631592, -2780.8664551, 2689.7343750
4: -740.1560059, 1722.6846924, -834.1444092, 1941.9635010, -2682.1191406, 2556.8286133

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7342120, upper bound: 1781.7344396
time: 0.50 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7342120, upper bound: 1781.7344901
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -360.3451538, 1456.6596680, -415.6548462, 1678.7097168, -2039.0548096, 1872.3142090
1: -579.9312744, 1613.1932373, -668.5964355, 1859.9970703, -2439.9282227, 2281.7893066
2: -433.1715393, 1856.8184814, -499.2538452, 2140.1540527, -2573.3256836, 2356.0722656
3: -931.8484497, 1662.3767090, -1075.7971191, 1915.3278809, -2847.1762695, 2738.1735840
4: -747.1314697, 1737.8078613, -860.6717529, 2002.8348389, -2749.9663086, 2598.4794922

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_B2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7335734, upper bound: 1781.7341589
time: 0.55 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_B2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7343226, upper bound: 1781.7345545
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -402.8233032, 1628.8464355, -357.1508179, 1444.1069336, -1846.9301758, 1985.9970703
1: -647.6110229, 1804.3489990, -575.0017700, 1599.2917480, -2246.9028320, 2379.3508301
2: -483.7769470, 2076.2917480, -429.3719788, 1840.7468262, -2324.5236816, 2505.6635742
3: -1042.1881104, 1856.8631592, -924.0037231, 1647.5461426, -2689.7343750, 2780.8664551
4: -834.1444092, 1941.9635010, -740.1560059, 1722.6846924, -2556.8286133, 2682.1191406

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7341673, upper bound: 1781.7341673
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7341673, upper bound: 1781.7341761
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -415.6548462, 1678.7097168, -360.3451538, 1456.6596680, -1872.3142090, 2039.0548096
1: -668.5964355, 1859.9970703, -579.9312744, 1613.1932373, -2281.7895508, 2439.9282227
2: -499.2538452, 2140.1540527, -433.1715393, 1856.8184814, -2356.0722656, 2573.3256836
3: -1075.7971191, 1915.3278809, -931.8484497, 1662.3767090, -2738.1735840, 2847.1762695
4: -860.6717529, 2002.8348389, -747.1314697, 1737.8078613, -2598.4794922, 2749.9663086

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7335518, upper bound: 1781.7335116
time: 0.56 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7342648, upper bound: 1781.7342612
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -407.2990723, 1647.0445557, -402.8233032, 1628.8464355, -2036.1451416, 2049.8679199
1: -654.7455444, 1824.4360352, -647.6110229, 1804.3489990, -2459.0944824, 2472.0466309
2: -489.1272583, 2099.4880371, -483.7769470, 2076.2917480, -2565.4189453, 2583.2648926
3: -1053.5203857, 1877.8547363, -1042.1881104, 1856.8631592, -2910.3835449, 2920.0424805
4: -843.5029907, 1963.7203369, -834.1444092, 1941.9635010, -2785.4660645, 2797.8645020

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7344845, upper bound: 1781.7344846
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7344845, upper bound: 1781.7345447
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -411.2431335, 1662.9313965, -415.6548462, 1678.7097168, -2089.9528809, 2078.5861816
1: -660.9100342, 1841.9162598, -668.5964355, 1859.9970703, -2520.9069824, 2510.5124512
2: -493.8054810, 2119.7707520, -499.2538452, 2140.1540527, -2633.9592285, 2619.0244141
3: -1063.3881836, 1896.3376465, -1075.7971191, 1915.3278809, -2978.7160645, 2972.1347656
4: -851.9455566, 1982.7741699, -860.6717529, 2002.8348389, -2854.7802734, 2843.4458008

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7345409, upper bound: 1781.7345027
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7345409, upper bound: 1781.7346113
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -365.0088196, 1475.8829346, -458.0041504, 1834.3841553, -2199.3930664, 1933.8870850
1: -587.5324097, 1634.4114990, -740.2851562, 2037.1063232, -2624.6381836, 2374.6965332
2: -438.8264771, 1881.2207031, -553.6544189, 2341.1132812, -2779.9396973, 2434.8747559
3: -943.7613525, 1684.4095459, -1191.9805908, 2102.5983887, -3046.3598633, 2875.0727539
4: -756.7219849, 1760.6367188, -953.4595947, 2195.0305176, -2951.7521973, 2714.0961914

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7118629, upper bound: 1781.7072166
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7118629, upper bound: 1781.7072166
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -364.8570862, 1475.2624512, -457.3205872, 1831.5590820, -2196.4162598, 1932.5830078
1: -587.2808228, 1633.7027588, -739.1828613, 2034.0225830, -2621.3034668, 2372.8857422
2: -438.6411133, 1880.4327393, -552.8548584, 2337.5122070, -2776.1530762, 2433.2873535
3: -943.3609009, 1683.6901855, -1190.2056885, 2099.4165039, -3042.7773438, 2872.6599121
4: -756.4165039, 1759.9118652, -952.0774536, 2191.6577148, -2948.0742188, 2711.9892578

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7119011, upper bound: 1781.7073405
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7119011, upper bound: 1781.7073405
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -415.0332947, 1678.4588623, -459.4661255, 1840.3011475, -2255.3344727, 2137.9250488
1: -667.1342163, 1859.1914062, -742.6494141, 2043.6488037, -2710.7829590, 2601.8405762
2: -498.3945312, 2139.5078125, -555.4129028, 2348.6689453, -2847.0632324, 2694.9201660
3: -1073.2490234, 1914.1794434, -1195.7912598, 2109.3376465, -3182.5866699, 3107.7946777
4: -859.6947021, 2001.2054443, -956.4998169, 2202.0993652, -3061.7939453, 2957.7050781

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_B1_A2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7041902, upper bound: 1781.7034770
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A1_B2_B1_A2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7041902, upper bound: 1781.7058771
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -413.8032227, 1673.6517334, -459.2172852, 1839.2855225, -2253.0886230, 2132.8691406
1: -665.5684814, 1853.7893066, -742.2589111, 2042.5541992, -2708.1225586, 2596.0478516
2: -497.0430908, 2133.4689941, -555.1142578, 2347.3725586, -2844.4152832, 2688.5832520
3: -1070.9802246, 1908.5093994, -1195.1795654, 2108.1743164, -3179.1538086, 3101.7084961
4: -857.1394043, 1995.6729736, -955.9790039, 2200.8857422, -3058.0249023, 2951.6516113

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_B1_A2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7041902, upper bound: 1781.7034770
time: 0.52 seconds

## Relational analysis of NS_A2_B2_A1_B2_B1_A2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7041902, upper bound: 1781.7058771
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -365.0088196, 1475.8829346, -480.3002014, 1927.8143311, -2292.8232422, 1956.1829834
1: -587.5324097, 1634.4114990, -774.6972046, 2139.8459473, -2727.3784180, 2409.1081543
2: -438.8264771, 1881.2207031, -579.6039429, 2459.3994141, -2898.2258301, 2460.8247070
3: -943.7613525, 1684.4095459, -1247.5859375, 2208.1982422, -3151.9594727, 2931.7604980
4: -756.7219849, 1760.6367188, -998.7920532, 2304.6918945, -3061.4138184, 2759.4284668

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7118758, upper bound: 1781.7071787
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7118758, upper bound: 1781.7071787
time: 0.54 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -364.8570862, 1475.2624512, -478.7261353, 1921.3687744, -2286.2258301, 1953.9885254
1: -587.2808228, 1633.7027588, -772.1367798, 2132.7263184, -2720.0070801, 2405.8391113
2: -438.6411133, 1880.4327393, -577.7418213, 2451.1293945, -2889.7705078, 2458.1745605
3: -943.3609009, 1683.6901855, -1243.5155029, 2200.9172363, -3144.2778320, 2927.0373535
4: -756.4165039, 1759.9118652, -995.5452271, 2296.9990234, -3053.4155273, 2755.4565430

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B2_B2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A1_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7085169, upper bound: 1781.7046771
time: 0.52 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_A1_B2_B2

### Relational analysis result of NS_A2_B2_A1_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7117849, upper bound: 1781.7075232
time: 0.50 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -416.3873901, 1683.9038086, -480.3002014, 1927.8143311, -2344.2014160, 2164.2038574
1: -669.3068237, 1865.1862793, -774.6972046, 2139.8459473, -2809.1528320, 2639.8835449
2: -500.0305176, 2146.4509277, -579.6039429, 2459.3994141, -2959.4299316, 2726.0549316
3: -1076.7180176, 1920.4570312, -1247.5859375, 2208.1982422, -3284.9162598, 3167.3203125
4: -862.5338745, 2007.7608643, -998.7920532, 2304.6918945, -3167.2253418, 3006.5524902

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7044017, upper bound: 1781.7043639
time: 0.55 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7044017, upper bound: 1781.7043639
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -415.7962341, 1681.5053711, -478.7261353, 1921.3687744, -2337.1650391, 2160.2314453
1: -668.3223877, 1862.4814453, -772.1367798, 2132.7263184, -2801.0485840, 2634.6181641
2: -499.3233948, 2143.3764648, -577.7418213, 2451.1293945, -2950.4523926, 2721.1181641
3: -1075.1044922, 1917.7860107, -1243.5155029, 2200.9172363, -3276.0214844, 3160.6652832
4: -861.3416748, 2004.9328613, -995.5452271, 2296.9990234, -3158.3408203, 3000.4772949

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7132624, upper bound: 1781.7082764
time: 0.56 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7132624, upper bound: 1781.7082764
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A2_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -456.1513977, 1826.5603027, -359.0148315, 1451.9942627, -1908.1453857, 2185.5751953
1: -737.2529297, 2027.9628906, -577.8498535, 1607.8750000, -2345.1274414, 2605.8127441
2: -551.3972778, 2331.6235352, -431.6076660, 1850.8400879, -2402.2373047, 2763.2312012
3: -1186.9655762, 2091.9506836, -928.3140869, 1656.7128906, -2842.0312500, 3020.2646484
4: -949.1479492, 2185.1174316, -744.3224487, 1731.8765869, -2681.0241699, 2929.4392090

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A2_B2_A2_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -457.9509888, 1833.9626465, -363.9831543, 1471.8001709, -1929.7512207, 2197.9458008
1: -740.2706299, 2036.7138672, -585.9177246, 1629.8797607, -2370.1501465, 2622.6315918
2: -553.5969238, 2340.6606445, -437.5832825, 1876.0383301, -2429.6352539, 2778.2438965
3: -1192.0166016, 2102.0070801, -941.2084961, 1679.6021729, -2870.1188965, 3043.2155762
4: -953.3206177, 2194.5327148, -754.5682983, 1755.6958008, -2709.0161133, 2949.1010742

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_A1_B1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7061561, upper bound: 1781.7066100
time: 0.53 seconds

## Relational analysis of NS_A2_B2_A2_A1_B1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7220764, upper bound: 1781.7238757
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A2_A1_B1_B2_B1

### Backsubstitution after applying NS history:
0: -459.4661255, 1840.3011475, -458.0041504, 1834.3841553, -2293.8503418, 2298.3051758
1: -742.6494141, 2043.6488037, -740.2851562, 2037.1063232, -2779.7553711, 2783.9340820
2: -555.4129028, 2348.6689453, -553.6544189, 2341.1132812, -2896.5258789, 2902.3229980
3: -1195.7912598, 2109.3376465, -1191.9805908, 2102.5983887, -3295.5319824, 3298.4577637
4: -956.4998169, 2202.0993652, -953.4595947, 2195.0305176, -3151.5295410, 3155.5590820

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_A1_B1_B2_B1_A1

### Relational analysis result of NS_A2_B2_A2_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7051064, upper bound: 1781.7051064
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A2_A1_B1_B2_B1_A2

### Relational analysis result of NS_A2_B2_A2_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7051064, upper bound: 1781.7051064
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A2_A1_B1_B2_B2

### Backsubstitution after applying NS history:
0: -459.2172852, 1839.2855225, -457.3205872, 1831.5590820, -2290.7763672, 2296.6062012
1: -742.2589111, 2042.5541992, -739.1828613, 2034.0225830, -2776.2814941, 2781.7370605
2: -555.1142578, 2347.3725586, -552.8548584, 2337.5122070, -2892.6264648, 2900.2272949
3: -1195.1795654, 2108.1743164, -1190.2056885, 2099.4165039, -3291.9211426, 3295.6103516
4: -955.9790039, 2200.8857422, -952.0774536, 2191.6577148, -3147.6364746, 3152.9631348

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_A1_B1_B2_B2_A1

### Relational analysis result of NS_A2_B2_A2_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7051064, upper bound: 1781.7051064
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A2_A1_B1_B2_B2_A2

### Relational analysis result of NS_A2_B2_A2_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7051064, upper bound: 1781.7051064
time: 0.52 seconds

## BFS NS instance: NS_A2_B2_A2_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -459.4661255, 1840.3011475, -412.8540344, 1670.0659180, -2129.5319824, 2253.1547852
1: -742.6494141, 2043.6488037, -663.7639771, 1849.8055420, -2592.4545898, 2707.4123535
2: -555.4129028, 2348.6689453, -495.7972107, 2128.8229980, -2684.2358398, 2844.4660645
3: -1195.7912598, 2109.3376465, -1067.7817383, 1904.3890381, -3098.0126953, 3177.1193848
4: -956.4998169, 2202.0993652, -855.2424927, 1990.9984131, -2947.4982910, 3057.3417969

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_A1_B2_B1_B1_A1

### Relational analysis result of NS_A2_B2_A2_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7090651, upper bound: 1781.7151797
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A2_A1_B2_B1_B1_A2

### Relational analysis result of NS_A2_B2_A2_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7090651, upper bound: 1781.7151797
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A2_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -459.2172852, 1839.2855225, -410.8474731, 1662.0363770, -2121.2536621, 2250.1325684
1: -742.2589111, 2042.5541992, -660.9581909, 1840.8428955, -2583.1018066, 2703.5124512
2: -555.1142578, 2347.3725586, -493.5069275, 2118.6677246, -2673.7819824, 2840.8793945
3: -1195.1795654, 2108.1743164, -1063.4946289, 1895.1348877, -3088.3498535, 3171.6689453
4: -955.9790039, 2200.8857422, -851.0590210, 1981.7194824, -2937.6979980, 3051.9445801

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_A1_B2_B1_B2_A1

### Relational analysis result of NS_A2_B2_A2_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7090651, upper bound: 1781.7151797
time: 0.53 seconds

## Relational analysis of NS_A2_B2_A2_A1_B2_B1_B2_A2

### Relational analysis result of NS_A2_B2_A2_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7090651, upper bound: 1781.7151797
time: 0.51 seconds

## BFS NS instance: NS_A2_B2_A2_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -459.4661255, 1840.3011475, -480.2918091, 1927.7802734, -2387.2463379, 2320.5930176
1: -742.6494141, 2043.6488037, -774.6840820, 2139.8083496, -2882.4575195, 2818.3330078
2: -555.4129028, 2348.6689453, -579.5936890, 2459.3562012, -3014.7687988, 2928.2626953
3: -1195.7912598, 2109.3376465, -1247.5640869, 2208.1606445, -3400.7155762, 3355.1240234
4: -956.4998169, 2202.0993652, -998.7745972, 2304.6516113, -3261.1513672, 3200.8737793

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_A1_B2_B2_B1_A1

### Relational analysis result of NS_A2_B2_A2_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7048083, upper bound: 1781.7047341
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A2_A1_B2_B2_B1_A2

### Relational analysis result of NS_A2_B2_A2_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7048083, upper bound: 1781.7047341
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A2_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -459.2172852, 1839.2855225, -478.5549011, 1920.6800537, -2379.8974609, 2317.8403320
1: -742.2589111, 2042.5541992, -771.8651123, 2131.9604492, -2874.2192383, 2814.4194336
2: -555.1142578, 2347.3725586, -577.5330811, 2450.2495117, -3005.3637695, 2924.9052734
3: -1195.1795654, 2108.1743164, -1243.0682373, 2200.1401367, -3392.2651367, 3349.5419922
4: -955.9790039, 2200.8857422, -995.1895752, 2296.1826172, -3252.1613770, 3196.0751953

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_A1_B2_B2_B2_A1

### Relational analysis result of NS_A2_B2_A2_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7049642, upper bound: 1781.7051994
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A2_A1_B2_B2_B2_A2

### Relational analysis result of NS_A2_B2_A2_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7049642, upper bound: 1781.7051994
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -480.3002014, 1927.8143311, -363.9963379, 1471.9134521, -1952.2136230, 2291.8105469
1: -774.6972046, 2139.8459473, -585.9166260, 1629.9927979, -2404.6899414, 2725.7624512
2: -579.6039429, 2459.3994141, -437.6012573, 1876.1569824, -2455.7609863, 2897.0007324
3: -1247.5859375, 2208.1982422, -941.1716919, 1679.7885742, -2927.1430664, 3149.3698730
4: -998.7920532, 2304.6918945, -754.6205444, 1755.8084717, -2754.6005859, 3059.3125000

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_A2_B1_A1_B1_B1

### Relational analysis result of NS_A2_B2_A2_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7034770, upper bound: 1781.7034993
time: 0.50 seconds

## Relational analysis of NS_A2_B2_A2_A2_B1_A1_B1_B2

### Relational analysis result of NS_A2_B2_A2_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7034770, upper bound: 1781.7037621
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -480.3002014, 1927.8143311, -459.4661255, 1840.3011475, -2320.6013184, 2387.2805176
1: -774.6972046, 2139.8459473, -742.6494141, 2043.6488037, -2818.3459473, 2882.4953613
2: -579.6039429, 2459.3994141, -555.4129028, 2348.6689453, -2928.2729492, 3014.8120117
3: -1247.5859375, 2208.1982422, -1195.7912598, 2109.3376465, -3355.1457520, 3400.7534180
4: -998.7920532, 2304.6918945, -956.4998169, 2202.0993652, -3200.8913574, 3261.1916504

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_A2_B1_A1_B2_B1

### Relational analysis result of NS_A2_B2_A2_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7034770, upper bound: 1781.7034993
time: 0.56 seconds

## Relational analysis of NS_A2_B2_A2_A2_B1_A1_B2_B2

### Relational analysis result of NS_A2_B2_A2_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7034770, upper bound: 1781.7037621
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -478.7261353, 1921.3687744, -363.8751831, 1471.4132080, -1950.1394043, 2285.2438965
1: -772.1367798, 2132.7263184, -585.7136230, 1629.4180908, -2401.5541992, 2718.4396973
2: -577.7418213, 2451.1293945, -437.4528198, 1875.5222168, -2453.2641602, 2888.5817871
3: -1243.5155029, 2200.9172363, -940.8489990, 1679.2086182, -2922.5600586, 3141.7653809
4: -995.5452271, 2296.9990234, -754.3786011, 1755.2290039, -2750.7741699, 3051.3776855

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1781.7007454, upper bound: 1781.7021412
time: 0.51 seconds

## Relational analysis of NS_A2_B2_A2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1781.7036655, upper bound: 1781.7036655
time: 0.56 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.81 + 418.24 = 421.05 seconds
