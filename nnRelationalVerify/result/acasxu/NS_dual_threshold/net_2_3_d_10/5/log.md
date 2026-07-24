## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_3.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 1783.0300611210841


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-444.5947571, 1548.2216797, -444.5947571, 1548.2216797, -1992.8161621, 1992.8161621)
1: (-450.9839478, 972.3717651, -450.9839478, 972.3717651, -1423.3557129, 1423.3557129)
2: (-411.3054810, 953.4530640, -411.3054810, 953.4530640, -1364.7585449, 1364.7585449)
3: (-487.6748962, 1164.5722656, -487.6748962, 1164.5722656, -1652.2471924, 1652.2471924)
4: (-566.0885620, 1053.3640137, -566.0885620, 1053.3640137, -1619.4525146, 1619.4525146)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.61 + 2.69 = 4.29 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1783.0478916, upper bound: 1783.0478916

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0430258, upper bound: 1783.0377310
time: 1.07 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0373774, upper bound: 1783.0373774
time: 0.97 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.19 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 2.19
Output dim: 0, lower bound: -1783.0430258, upper bound: 1783.0377310
NS_B2, status: Status.UNKNOWN, split count: 1, time: 2.19
Output dim: 0, lower bound: -1783.0373774, upper bound: 1783.0373774

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -444.5947571, 1548.2216797, -428.2842102, 1491.9852295, -1936.5798340, 1976.5057373
1: -450.9839478, 972.3717651, -434.5842590, 937.1694946, -1388.1534424, 1406.9560547
2: -411.3054810, 953.4530640, -396.5590820, 919.0173950, -1330.3228760, 1350.0120850
3: -487.6748962, 1164.5722656, -469.8634644, 1122.2145996, -1609.8895264, 1634.4357910
4: -566.0885620, 1053.3640137, -545.6523438, 1015.0709839, -1581.1595459, 1599.0163574

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0373774, upper bound: 1783.0373774
time: 0.81 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0373774, upper bound: 1783.0373774
time: 0.85 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -437.3733521, 1523.9152832, -594.3931274, 2079.2519531, -2515.3498535, 2118.3083496
1: -443.6616211, 957.0374146, -604.2772217, 1305.1914062, -1748.8530273, 1560.7330322
2: -404.9277954, 938.4420776, -553.0122681, 1282.3945312, -1686.8040771, 1491.4543457
3: -479.7664185, 1146.1600342, -653.8726807, 1561.0693359, -2040.8356934, 1798.4418945
4: -557.2370605, 1036.5720215, -760.8148193, 1413.2551270, -1970.2348633, 1797.3868408

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0373774, upper bound: 1783.0373774
time: 1.07 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0373774, upper bound: 1783.0373774
time: 0.94 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.67 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 3.67
Output dim: 0, lower bound: -1783.0373774, upper bound: 1783.0373774
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 3.67
Output dim: 0, lower bound: -1783.0373774, upper bound: 1783.0373774
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 3.67
Output dim: 0, lower bound: -1783.0373774, upper bound: 1783.0373774
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 3.67
Output dim: 0, lower bound: -1783.0373774, upper bound: 1783.0373774

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -428.2842102, 1491.9852295, -428.2842102, 1491.9852295, -1920.2694092, 1920.2694092
1: -434.5842590, 937.1694946, -434.5842590, 937.1694946, -1371.7537842, 1371.7537842
2: -396.5590820, 919.0173950, -396.5590820, 919.0173950, -1315.5764160, 1315.5764160
3: -469.8634644, 1122.2145996, -469.8634644, 1122.2145996, -1592.0781250, 1592.0781250
4: -545.6523438, 1015.0709839, -545.6523438, 1015.0709839, -1560.7233887, 1560.7233887

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_B1

### Relational analysis result of NS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0410776, upper bound: 1783.0365320
time: 0.87 seconds

## Relational analysis of NS_B1_A1_B2

### Relational analysis result of NS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0383144, upper bound: 1783.0363150
time: 0.79 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -594.3806763, 2079.2126465, -428.2842102, 1491.9852295, -2086.3652344, 2506.2180176
1: -604.2645264, 1305.1657715, -434.5842590, 937.1694946, -1540.8405762, 1739.7500000
2: -553.0017090, 1282.3690186, -396.5590820, 919.0173950, -1472.0190430, 1678.4050293
3: -653.8582153, 1561.0386963, -469.8634644, 1122.2145996, -1774.4677734, 2030.9020996
4: -760.8002319, 1413.2255859, -545.6523438, 1015.0709839, -1775.8712158, 1958.6104736

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0392175, upper bound: 1783.0350540
time: 0.82 seconds

## Relational analysis of NS_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0383833, upper bound: 1783.0350129
time: 1.23 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -428.2842102, 1491.9852295, -594.3806763, 2079.2126465, -2506.2180176, 2086.3654785
1: -434.5842590, 937.1694946, -604.2645264, 1305.1657715, -1739.7500000, 1540.8405762
2: -396.5590820, 919.0173950, -553.0017090, 1282.3690186, -1678.4050293, 1472.0190430
3: -469.8634644, 1122.2145996, -653.8582153, 1561.0386963, -2030.9020996, 1774.4677734
4: -545.6523438, 1015.0709839, -760.8002319, 1413.2255859, -1958.6104736, 1775.8712158

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_A1_A1

### Relational analysis result of NS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0362999, upper bound: 1783.0372802
time: 0.78 seconds

## Relational analysis of NS_B2_A1_A2

### Relational analysis result of NS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0362598, upper bound: 1783.0362598
time: 1.02 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -594.3931274, 2079.2519531, -594.3931274, 2079.2519531, -2671.8896484, 2671.8896484
1: -604.2772217, 1305.1914062, -604.2772217, 1305.1914062, -1907.9316406, 1907.9316406
2: -553.0122681, 1282.3945312, -553.0122681, 1282.3945312, -1833.8972168, 1833.8972168
3: -653.8726807, 1561.0693359, -653.8726807, 1561.0693359, -2211.6166992, 2211.6166992
4: -760.8148193, 1413.2551270, -760.8148193, 1413.2551270, -2173.5646973, 2173.5646973

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_A2_A1

### Relational analysis result of NS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0362999, upper bound: 1783.0372150
time: 1.06 seconds

## Relational analysis of NS_B2_A2_A2

### Relational analysis result of NS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0362598, upper bound: 1783.0362598
time: 0.76 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.49 seconds
NS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.49
Output dim: 0, lower bound: -1783.0410776, upper bound: 1783.0365320
NS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.49
Output dim: 0, lower bound: -1783.0383144, upper bound: 1783.0363150
NS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.49
Output dim: 0, lower bound: -1783.0392175, upper bound: 1783.0350540
NS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.49
Output dim: 0, lower bound: -1783.0383833, upper bound: 1783.0350129
NS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 3.49
Output dim: 0, lower bound: -1783.0362999, upper bound: 1783.0372802
NS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 3.49
Output dim: 0, lower bound: -1783.0362598, upper bound: 1783.0362598
NS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 3.49
Output dim: 0, lower bound: -1783.0362999, upper bound: 1783.0372150
NS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.49
Output dim: 0, lower bound: -1783.0362598, upper bound: 1783.0362598

## BFS NS instance: NS_B1_A1_B1

### Backsubstitution after applying NS history:
0: -428.2842102, 1491.9852295, -425.9190979, 1483.8929443, -1912.1770020, 1917.9039307
1: -434.5842590, 937.1694946, -432.2026367, 932.0657349, -1366.6500244, 1369.3720703
2: -396.5590820, 919.0173950, -394.4095154, 914.0342407, -1310.5930176, 1313.4267578
3: -469.8634644, 1122.2145996, -467.2754822, 1116.0883789, -1585.9519043, 1589.4899902
4: -545.6523438, 1015.0709839, -542.6821289, 1009.5067749, -1555.1589355, 1557.7530518

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_A1_B1_A1

### Relational analysis result of NS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0390830, upper bound: 1783.0374212
time: 1.24 seconds

## Relational analysis of NS_B1_A1_B1_A2

### Relational analysis result of NS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0406339, upper bound: 1783.0385269
time: 0.97 seconds

## BFS NS instance: NS_B1_A1_B2

### Backsubstitution after applying NS history:
0: -425.0173340, 1480.1507568, -529.0075684, 1830.6269531, -2255.6442871, 2009.1583252
1: -431.2369690, 929.6313477, -536.9533081, 1147.5079346, -1578.7448730, 1466.5847168
2: -393.4515686, 911.5681763, -484.9090881, 1125.9398193, -1519.3913574, 1396.4772949
3: -466.2825012, 1113.3392334, -580.1031494, 1376.1933594, -1842.4758301, 1693.4423828
4: -541.3381958, 1007.0415039, -668.1859741, 1248.2287598, -1789.5668945, 1675.2274170

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_B2_A1

### Relational analysis result of NS_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0383799, upper bound: 1783.0383799
time: 0.99 seconds

## Relational analysis of NS_B1_A1_B2_A2

### Relational analysis result of NS_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0383799, upper bound: 1783.0383799
time: 0.91 seconds

## BFS NS instance: NS_B1_A2_B1

### Backsubstitution after applying NS history:
0: -580.7532349, 2032.6069336, -397.4479065, 1386.0675049, -1966.8208008, 2428.5793457
1: -590.5700073, 1276.0992432, -403.6125793, 870.8452759, -1460.7993164, 1679.7117920
2: -540.7680054, 1253.9661865, -368.4636230, 854.3245239, -1395.0924072, 1621.8688965
3: -638.9947510, 1525.8597412, -436.1403198, 1042.2551270, -1679.5336914, 1962.0000000
4: -743.8869629, 1381.5040283, -506.9649658, 943.0471802, -1686.9340820, 1888.1268311

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B1_B1

### Relational analysis result of NS_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0380152, upper bound: 1783.0343802
time: 1.06 seconds

## Relational analysis of NS_B1_A2_B1_B2

### Relational analysis result of NS_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0390638, upper bound: 1783.0349838
time: 0.86 seconds

## BFS NS instance: NS_B1_A2_B2

### Backsubstitution after applying NS history:
0: -567.8981934, 1989.3281250, -415.3688660, 1454.3197021, -2022.2178955, 2403.5185547
1: -577.1441040, 1249.1700439, -421.1046143, 913.0626221, -1489.6364746, 1670.2745361
2: -529.2065430, 1227.3508301, -386.2360840, 896.6438599, -1425.8502197, 1613.1176758
3: -624.7398071, 1493.5417480, -455.6521606, 1092.4167480, -1715.3941650, 1949.1937256
4: -728.1588745, 1351.1420898, -531.7120361, 987.4636230, -1715.6223145, 1882.6643066

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B2_B1

### Relational analysis result of NS_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0377104, upper bound: 1783.0343798
time: 1.46 seconds

## Relational analysis of NS_B1_A2_B2_B2

### Relational analysis result of NS_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0382893, upper bound: 1783.0349618
time: 0.94 seconds

## BFS NS instance: NS_B2_A1_A1

### Backsubstitution after applying NS history:
0: -425.9190979, 1483.8929443, -594.3801270, 2079.2109375, -2503.8493652, 2078.2729492
1: -432.2026367, 932.0657349, -604.2640991, 1305.1647949, -1737.3674316, 1535.7304688
2: -394.4095154, 914.0342407, -553.0012817, 1282.3677979, -1676.2529297, 1467.0355225
3: -467.2754822, 1116.0883789, -653.8575439, 1561.0375977, -2028.3129883, 1768.3314209
4: -542.6821289, 1009.5067749, -760.7996216, 1413.2243652, -1955.6357422, 1770.3063965

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_A1_B1

### Relational analysis result of NS_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0262983, upper bound: 1783.0360921
time: 0.66 seconds

## Relational analysis of NS_B2_A1_A1_B2

### Relational analysis result of NS_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0320555, upper bound: 1783.0383138
time: 1.33 seconds

## BFS NS instance: NS_B2_A1_A2

### Backsubstitution after applying NS history:
0: -528.7788696, 1829.8413086, -590.4616089, 2065.4521484, -2592.2819824, 2420.3027344
1: -536.7155762, 1147.0328369, -600.3024292, 1296.2806396, -1832.9962158, 1746.4724121
2: -484.7057495, 1125.4658203, -549.2724609, 1273.6712646, -1757.6636963, 1674.7382812
3: -579.8515015, 1375.6126709, -649.5740967, 1550.5544434, -2130.0405273, 2023.4887695
4: -667.9004517, 1247.6911621, -755.6986694, 1403.7438965, -2071.0278320, 2003.3898926

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B2_A1_A2_B1

### Relational analysis result of NS_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0363150, upper bound: 1783.0383144
time: 0.85 seconds

## Relational analysis of NS_B2_A1_A2_B2

### Relational analysis result of NS_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0363150, upper bound: 1783.0383144
time: 0.87 seconds

## BFS NS instance: NS_B2_A2_A1

### Backsubstitution after applying NS history:
0: -592.4514160, 2072.6411133, -594.3931274, 2079.2519531, -2669.9443359, 2665.2543945
1: -602.3081665, 1301.0325928, -604.2772217, 1305.1914062, -1905.9543457, 1903.7600098
2: -551.2530518, 1278.3327637, -553.0122681, 1282.3945312, -1832.1326904, 1829.8222656
3: -651.7415161, 1556.0515137, -653.8726807, 1561.0693359, -2209.4719238, 2206.5839844
4: -758.3903198, 1408.7135010, -760.8148193, 1413.2551270, -2171.1362305, 2169.0107422

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B2_A2_A1_B1

### Relational analysis result of NS_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0362598, upper bound: 1783.0362598
time: 0.80 seconds

## Relational analysis of NS_B2_A2_A1_B2

### Relational analysis result of NS_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0362598, upper bound: 1783.0362598
time: 1.02 seconds

## BFS NS instance: NS_B2_A2_A2

### Backsubstitution after applying NS history:
0: -678.0094604, 2362.3596191, -590.4718018, 2065.4846191, -2741.1232910, 2950.7019043
1: -689.8403931, 1478.6501465, -600.3127441, 1296.3012695, -1983.5340576, 2077.4174805
2: -626.1655884, 1453.7469482, -549.2810669, 1273.6920166, -1898.2955322, 2001.3227539
3: -745.4949951, 1771.5964355, -649.5859375, 1550.5794678, -2291.6479492, 2417.7636719
4: -862.7015381, 1607.3531494, -755.7105103, 1403.7678223, -2265.6166992, 2361.6254883

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B2_A2_A2_B1

### Relational analysis result of NS_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0362598, upper bound: 1783.0362598
time: 0.85 seconds

## Relational analysis of NS_B2_A2_A2_B2

### Relational analysis result of NS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0362598, upper bound: 1783.0362598
time: 0.80 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.34 seconds
NS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 0, lower bound: -1783.0390830, upper bound: 1783.0374212
NS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 0, lower bound: -1783.0406339, upper bound: 1783.0385269
NS_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 0, lower bound: -1783.0383799, upper bound: 1783.0383799
NS_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 0, lower bound: -1783.0383799, upper bound: 1783.0383799
NS_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 0, lower bound: -1783.0380152, upper bound: 1783.0343802
NS_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 0, lower bound: -1783.0390638, upper bound: 1783.0349838
NS_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 0, lower bound: -1783.0377104, upper bound: 1783.0343798
NS_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 0, lower bound: -1783.0382893, upper bound: 1783.0349618
NS_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 0, lower bound: -1783.0262983, upper bound: 1783.0360921
NS_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 0, lower bound: -1783.0320555, upper bound: 1783.0383138
NS_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 0, lower bound: -1783.0363150, upper bound: 1783.0383144
NS_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 0, lower bound: -1783.0363150, upper bound: 1783.0383144
NS_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 0, lower bound: -1783.0362598, upper bound: 1783.0362598
NS_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 0, lower bound: -1783.0362598, upper bound: 1783.0362598
NS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 0, lower bound: -1783.0362598, upper bound: 1783.0362598
NS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.34
Output dim: 0, lower bound: -1783.0362598, upper bound: 1783.0362598

## BFS NS instance: NS_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -404.2165222, 1406.4442139, -418.0602112, 1455.9578857, -1860.1744385, 1824.5042725
1: -410.0805054, 884.2559814, -424.1939087, 914.7949219, -1324.8752441, 1308.4498291
2: -374.0447388, 866.6723633, -387.0720520, 896.9453735, -1270.9897461, 1253.7442627
3: -443.4322205, 1058.6242676, -458.6324158, 1095.3281250, -1538.7603760, 1517.2565918
4: -514.4911499, 957.3106689, -532.5267334, 990.6358032, -1505.1268311, 1489.8370361

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B1_A1_B1_A1_B1

### Relational analysis result of NS_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0390830, upper bound: 1783.0374212
time: 0.82 seconds

## Relational analysis of NS_B1_A1_B1_A1_B2

### Relational analysis result of NS_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0390830, upper bound: 1783.0374212
time: 0.80 seconds

## BFS NS instance: NS_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -493.2639771, 1705.1289062, -410.6025696, 1428.8769531, -1922.1408691, 2115.7312012
1: -499.6026306, 1072.6654053, -416.3334045, 897.3383179, -1396.9409180, 1488.9987793
2: -453.1403198, 1048.3687744, -379.4108276, 878.8862915, -1332.0266113, 1427.7794189
3: -542.2065430, 1285.1954346, -450.5617676, 1074.4368896, -1616.6434326, 1735.7572021
4: -623.2268677, 1160.9147949, -522.2117310, 971.2323608, -1594.4592285, 1683.1264648

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B1_A1_B1_A2_B1

### Relational analysis result of NS_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0406339, upper bound: 1783.0385197
time: 0.83 seconds

## Relational analysis of NS_B1_A1_B1_A2_B2

### Relational analysis result of NS_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0406339, upper bound: 1783.0385269
time: 0.94 seconds

## BFS NS instance: NS_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -425.9190979, 1483.8929443, -528.9265137, 1830.3485107, -2256.2675781, 2012.8194580
1: -432.2026367, 932.0657349, -536.8690186, 1147.3395996, -1579.5422363, 1468.9348145
2: -394.4095154, 914.0342407, -484.8370056, 1125.7719727, -1520.1811523, 1398.8709717
3: -467.2754822, 1116.0883789, -580.0139771, 1375.9874268, -1843.2629395, 1696.1022949
4: -542.6821289, 1009.5067749, -668.0848389, 1248.0382080, -1790.7203369, 1677.5915527

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_A1_B2_A1_A1

### Relational analysis result of NS_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0380162, upper bound: 1783.0373407
time: 0.80 seconds

## Relational analysis of NS_B1_A1_B2_A1_A2

### Relational analysis result of NS_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0383466, upper bound: 1783.0383466
time: 0.75 seconds

## BFS NS instance: NS_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -529.0809326, 1830.8786621, -529.0809326, 1830.8786621, -2359.9592285, 2359.9594727
1: -537.0295410, 1147.6604004, -537.0295410, 1147.6604004, -1684.6899414, 1684.6899414
2: -484.9742737, 1126.0916748, -484.9742737, 1126.0916748, -1611.0657959, 1611.0657959
3: -580.1837769, 1376.3797607, -580.1837769, 1376.3797607, -1956.5634766, 1956.5634766
4: -668.2774048, 1248.4008789, -668.2774048, 1248.4008789, -1916.6782227, 1916.6782227

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_A1_B2_A2_A1

### Relational analysis result of NS_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0380162, upper bound: 1783.0373407
time: 0.94 seconds

## Relational analysis of NS_B1_A1_B2_A2_A2

### Relational analysis result of NS_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0383466, upper bound: 1783.0383466
time: 0.93 seconds

## BFS NS instance: NS_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -576.4319458, 2017.8476562, -379.2249451, 1325.4808350, -1901.9127197, 2395.4624023
1: -586.2414551, 1266.8912354, -385.2583923, 831.9400024, -1417.4829102, 1652.1496582
2: -536.8982544, 1244.9691162, -351.6981506, 816.9230957, -1353.8212891, 1596.0434570
3: -634.3042603, 1514.7137451, -416.1432495, 995.4196777, -1627.9259033, 1930.8569336
4: -738.5199585, 1371.4160156, -484.2012329, 900.9321899, -1639.4521484, 1855.2215576

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B1_B1_B1

### Relational analysis result of NS_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0369292, upper bound: 1783.0335432
time: 0.86 seconds

## Relational analysis of NS_B1_A2_B1_B1_B2

### Relational analysis result of NS_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0358852, upper bound: 1783.0334791
time: 0.90 seconds

## BFS NS instance: NS_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -561.2079468, 1969.9013672, -403.1340637, 1413.5269775, -1974.7348633, 2371.7800293
1: -570.6834106, 1234.4020996, -409.6786194, 884.8515625, -1455.1580811, 1644.0806885
2: -521.9948730, 1214.9655762, -373.5916443, 870.5139160, -1392.5087891, 1588.1281738
3: -617.3125610, 1475.7061768, -442.0341797, 1058.8632812, -1674.7226562, 1917.7403564
4: -719.3630371, 1337.0152588, -515.3617554, 959.2703247, -1678.6333008, 1852.1304932

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B1_B2_B1

### Relational analysis result of NS_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0378403, upper bound: 1783.0341853
time: 1.11 seconds

## Relational analysis of NS_B1_A2_B1_B2_B2

### Relational analysis result of NS_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0378295, upper bound: 1783.0341981
time: 0.88 seconds

## BFS NS instance: NS_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -563.3577881, 1973.8314209, -394.0091858, 1382.5845947, -1945.9423828, 2366.5212402
1: -572.5720215, 1239.4803467, -399.5766602, 867.4624634, -1439.3787842, 1639.0568848
2: -525.1415405, 1217.8741455, -366.7454834, 852.5574951, -1377.6988525, 1584.0855713
3: -619.7858276, 1481.8223877, -432.2330017, 1037.4185791, -1655.3635254, 1914.0554199
4: -722.5274048, 1340.5061035, -505.0932312, 937.8228760, -1660.3498535, 1845.3533936

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_B2_B1_B1

### Relational analysis result of NS_B1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0374267, upper bound: 1783.0339720
time: 0.78 seconds

## Relational analysis of NS_B1_A2_B2_B1_B2

### Relational analysis result of NS_B1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0354439, upper bound: 1783.0334879
time: 0.89 seconds

## BFS NS instance: NS_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -550.0369873, 1931.9550781, -419.7165527, 1479.5455322, -2029.5823975, 2350.7319336
1: -559.0623779, 1210.8489990, -426.3051147, 924.1716309, -1482.8891602, 1637.1540527
2: -511.8376160, 1191.6704102, -389.6384583, 910.7514648, -1422.5891113, 1580.9871826
3: -605.0197144, 1447.4381104, -460.4071655, 1105.2255859, -1708.7408447, 1907.8452148
4: -705.5222168, 1310.5484619, -538.0189819, 1000.9852295, -1706.5074463, 1848.4792480

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_B2_B2_B1

### Relational analysis result of NS_B1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0376279, upper bound: 1783.0340151
time: 0.85 seconds

## Relational analysis of NS_B1_A2_B2_B2_B2

### Relational analysis result of NS_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0355099, upper bound: 1783.0338304
time: 0.75 seconds

## BFS NS instance: NS_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -394.7507324, 1373.5118408, -560.3114014, 1959.8164062, -2352.2854004, 1933.8231201
1: -400.3732300, 862.9658813, -569.6444702, 1230.5491943, -1630.9223633, 1431.4090576
2: -365.4772034, 845.2134399, -521.7239990, 1208.6047363, -1573.1234131, 1366.9373779
3: -433.1559753, 1033.5815430, -616.5117188, 1471.5889893, -1904.7449951, 1647.7944336
4: -502.8408203, 934.2100220, -717.6134644, 1331.7619629, -1833.8947754, 1651.8234863

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_A1_B1_A1

### Relational analysis result of NS_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0261276, upper bound: 1783.0353871
time: 0.77 seconds

## Relational analysis of NS_B2_A1_A1_B1_A2

### Relational analysis result of NS_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0261276, upper bound: 1783.0360921
time: 1.17 seconds

## BFS NS instance: NS_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -419.5654297, 1462.0435791, -565.7089844, 1976.5646973, -2396.1301270, 2027.7525635
1: -425.7899475, 918.2283936, -574.6122437, 1240.9464111, -1666.7363281, 1492.8405762
2: -388.5558777, 900.5294800, -525.6024780, 1218.4370117, -1606.9929199, 1426.1319580
3: -460.3092651, 1099.4647217, -621.8958740, 1484.5954590, -1944.9047852, 1720.5986328
4: -534.6223145, 994.4929810, -723.2510376, 1343.2891846, -1877.9114990, 1717.7440186

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_A1_B2_A1

### Relational analysis result of NS_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0318301, upper bound: 1783.0366455
time: 0.87 seconds

## Relational analysis of NS_B2_A1_A1_B2_A2

### Relational analysis result of NS_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0318301, upper bound: 1783.0383138
time: 0.93 seconds

## BFS NS instance: NS_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -528.6882324, 1829.5294189, -592.4382324, 2072.5988770, -2599.2746582, 2421.9677734
1: -536.6212158, 1146.8439941, -602.2947998, 1301.0054932, -1837.6267090, 1748.2697754
2: -484.6250916, 1125.2781982, -551.2419434, 1278.3057861, -1762.1773682, 1676.5201416
3: -579.7515869, 1375.3822021, -651.7260742, 1556.0189209, -2135.3657227, 2025.3900146
4: -667.7870483, 1247.4779053, -758.3748779, 1408.6822510, -2075.8300781, 2005.8527832

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_A2_B1_B1

### Relational analysis result of NS_B2_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0167596, upper bound: 1783.0234776
time: 0.75 seconds

## Relational analysis of NS_B2_A1_A2_B1_B2

### Relational analysis result of NS_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0316596, upper bound: 1783.0319134
time: 1.02 seconds

## BFS NS instance: NS_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -529.0809326, 1830.8786621, -678.0094604, 2362.3596191, -2889.0688477, 2508.8879395
1: -537.0295410, 1147.6604004, -689.8403931, 1478.6501465, -2015.6796875, 1835.5467529
2: -484.9742737, 1126.0916748, -626.1655884, 1453.7469482, -1937.7695312, 1752.2572021
3: -580.1837769, 1376.3797607, -745.4949951, 1771.5964355, -2351.2873535, 2119.0417480
4: -668.2774048, 1248.4008789, -862.7015381, 1607.3531494, -2274.0622559, 2111.1022949

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_A2_B2_B1

### Relational analysis result of NS_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0167596, upper bound: 1783.0302567
time: 0.75 seconds

## Relational analysis of NS_B2_A1_A2_B2_B2

### Relational analysis result of NS_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0316596, upper bound: 1783.0319134
time: 0.70 seconds

## BFS NS instance: NS_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -592.4514160, 2072.6411133, -592.4514160, 2072.6411133, -2663.3093262, 2663.3093262
1: -602.3081665, 1301.0325928, -602.3081665, 1301.0325928, -1901.7827148, 1901.7827148
2: -551.2530518, 1278.3327637, -551.2530518, 1278.3327637, -1828.0576172, 1828.0577393
3: -651.7415161, 1556.0515137, -651.7415161, 1556.0515137, -2204.4392090, 2204.4394531
4: -758.3903198, 1408.7135010, -758.3903198, 1408.7135010, -2166.5820312, 2166.5820312

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A2_A1_B1_A1

### Relational analysis result of NS_B2_A2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0234774, upper bound: 1783.0167596
time: 0.73 seconds

## Relational analysis of NS_B2_A2_A1_B1_A2

### Relational analysis result of NS_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0319971, upper bound: 1783.0349768
time: 1.01 seconds

## BFS NS instance: NS_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -592.4514160, 2072.6411133, -677.9715576, 2362.2324219, -2952.5466309, 2748.1779785
1: -602.3081665, 1301.0325928, -689.8009644, 1478.5726318, -2079.3291016, 1988.1923828
2: -551.2530518, 1278.3327637, -626.1321411, 1453.6702881, -2003.1981201, 1902.8626709
3: -651.7415161, 1556.0515137, -745.4533081, 1771.5012207, -2419.8032227, 2297.0375977
4: -758.3903198, 1408.7135010, -862.6548462, 1607.2651367, -2364.2109375, 2270.4916992

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A2_A1_B2_A1

### Relational analysis result of NS_B2_A2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0234774, upper bound: 1783.0167596
time: 0.94 seconds

## Relational analysis of NS_B2_A2_A1_B2_A2

### Relational analysis result of NS_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0319971, upper bound: 1783.0349768
time: 0.93 seconds

## BFS NS instance: NS_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -677.9504395, 2362.1608887, -592.4514160, 2072.6411133, -2748.1572266, 2952.4750977
1: -689.7788086, 1478.5289307, -602.3081665, 1301.0325928, -1988.1705322, 2079.2856445
2: -626.1130981, 1453.6269531, -551.2530518, 1278.3327637, -1902.8437500, 2003.1547852
3: -745.4298706, 1771.4475098, -651.7415161, 1556.0515137, -2297.0146484, 2419.7495117
4: -862.6285400, 1607.2156982, -758.3903198, 1408.7135010, -2270.4655762, 2364.1613770

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_A2_A2_B1_B1

### Relational analysis result of NS_B2_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0167596, upper bound: 1783.0234774
time: 0.84 seconds

## Relational analysis of NS_B2_A2_A2_B1_B2

### Relational analysis result of NS_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0316596, upper bound: 1783.0316596
time: 0.74 seconds

## BFS NS instance: NS_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -678.0094604, 2362.3596191, -678.0094604, 2362.3596191, -3037.5793457, 3037.5793457
1: -689.8403931, 1478.6501465, -689.8403931, 1478.6501465, -2165.8552246, 2165.8552246
2: -626.1655884, 1453.7469482, -626.1655884, 1453.7469482, -2078.1132812, 2078.1130371
3: -745.4949951, 1771.5964355, -745.4949951, 1771.5964355, -2512.5385742, 2512.5385742
4: -862.7015381, 1607.3531494, -862.7015381, 1607.3531494, -2468.2548828, 2468.2546387

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A2_A2_B2_A1

### Relational analysis result of NS_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0312349, upper bound: 1783.0260053
time: 0.94 seconds

## Relational analysis of NS_B2_A2_A2_B2_A2

### Relational analysis result of NS_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0316596, upper bound: 1783.0316596
time: 0.68 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 6.20 seconds
NS_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 6.20
Output dim: 0, lower bound: -1783.0390830, upper bound: 1783.0374212
NS_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 6.20
Output dim: 0, lower bound: -1783.0390830, upper bound: 1783.0374212
NS_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 6.20
Output dim: 0, lower bound: -1783.0406339, upper bound: 1783.0385197
NS_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 6.20
Output dim: 0, lower bound: -1783.0406339, upper bound: 1783.0385269
NS_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 6.20
Output dim: 0, lower bound: -1783.0380162, upper bound: 1783.0373407
NS_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 6.20
Output dim: 0, lower bound: -1783.0383466, upper bound: 1783.0383466
NS_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 6.20
Output dim: 0, lower bound: -1783.0380162, upper bound: 1783.0373407
NS_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 6.20
Output dim: 0, lower bound: -1783.0383466, upper bound: 1783.0383466
NS_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 6.20
Output dim: 0, lower bound: -1783.0369292, upper bound: 1783.0335432
NS_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 6.20
Output dim: 0, lower bound: -1783.0358852, upper bound: 1783.0334791
NS_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 6.20
Output dim: 0, lower bound: -1783.0378403, upper bound: 1783.0341853
NS_B1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 6.20
Output dim: 0, lower bound: -1783.0378295, upper bound: 1783.0341981
NS_B1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 6.20
Output dim: 0, lower bound: -1783.0374267, upper bound: 1783.0339720
NS_B1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 6.20
Output dim: 0, lower bound: -1783.0354439, upper bound: 1783.0334879
NS_B1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 6.20
Output dim: 0, lower bound: -1783.0376279, upper bound: 1783.0340151
NS_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 6.20
Output dim: 0, lower bound: -1783.0355099, upper bound: 1783.0338304
NS_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.20
Output dim: 0, lower bound: -1783.0261276, upper bound: 1783.0353871
NS_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.20
Output dim: 0, lower bound: -1783.0261276, upper bound: 1783.0360921
NS_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.20
Output dim: 0, lower bound: -1783.0318301, upper bound: 1783.0366455
NS_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.20
Output dim: 0, lower bound: -1783.0318301, upper bound: 1783.0383138
NS_B2_A1_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 6.20
Output dim: 0, lower bound: -1783.0167596, upper bound: 1783.0234776
NS_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 6.20
Output dim: 0, lower bound: -1783.0316596, upper bound: 1783.0319134
NS_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 6.20
Output dim: 0, lower bound: -1783.0167596, upper bound: 1783.0302567
NS_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 6.20
Output dim: 0, lower bound: -1783.0316596, upper bound: 1783.0319134
NS_B2_A2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.20
Output dim: 0, lower bound: -1783.0234774, upper bound: 1783.0167596
NS_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.20
Output dim: 0, lower bound: -1783.0319971, upper bound: 1783.0349768
NS_B2_A2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 6.20
Output dim: 0, lower bound: -1783.0234774, upper bound: 1783.0167596
NS_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.20
Output dim: 0, lower bound: -1783.0319971, upper bound: 1783.0349768
NS_B2_A2_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 6.20
Output dim: 0, lower bound: -1783.0167596, upper bound: 1783.0234774
NS_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 6.20
Output dim: 0, lower bound: -1783.0316596, upper bound: 1783.0316596
NS_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.20
Output dim: 0, lower bound: -1783.0312349, upper bound: 1783.0260053
NS_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.20
Output dim: 0, lower bound: -1783.0316596, upper bound: 1783.0316596

## BFS NS instance: NS_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -404.2165222, 1406.4442139, -402.1864014, 1399.4794922, -1803.6960449, 1808.6306152
1: -410.0805054, 884.2559814, -408.0348816, 879.8641357, -1289.9444580, 1292.2908936
2: -374.0447388, 866.6723633, -372.1815186, 862.3933716, -1236.4378662, 1238.8536377
3: -443.4322205, 1058.6242676, -441.2138977, 1053.3394775, -1496.7717285, 1499.8381348
4: -514.4911499, 957.3106689, -511.9244995, 952.5472412, -1467.0383301, 1469.2349854

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0390830, upper bound: 1783.0374212
time: 0.90 seconds

## Relational analysis of NS_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0390830, upper bound: 1783.0374212
time: 0.94 seconds

## BFS NS instance: NS_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -404.2165222, 1406.4442139, -491.0123291, 1697.2462158, -2101.4626465, 1897.4565430
1: -410.0805054, 884.2559814, -497.3096619, 1067.7528076, -1477.8332520, 1381.5656738
2: -374.0447388, 866.6723633, -451.0885925, 1043.4934082, -1417.5378418, 1317.7606201
3: -443.4322205, 1058.6242676, -539.7470703, 1279.3251953, -1722.7574463, 1598.3713379
4: -514.4911499, 957.3106689, -620.3925171, 1155.5444336, -1670.0356445, 1577.7028809

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0390830, upper bound: 1783.0374212
time: 0.94 seconds

## Relational analysis of NS_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0390830, upper bound: 1783.0374212
time: 1.20 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -493.2639771, 1705.1289062, -402.1864014, 1399.4794922, -1892.7434082, 2107.3146973
1: -499.6026306, 1072.6654053, -408.0348816, 879.8641357, -1379.4667969, 1480.7003174
2: -453.1403198, 1048.3687744, -372.1815186, 862.3933716, -1315.5336914, 1420.5500488
3: -542.2065430, 1285.1954346, -441.2138977, 1053.3394775, -1595.5460205, 1726.4093018
4: -623.2268677, 1160.9147949, -511.9244995, 952.5472412, -1575.7740479, 1672.8393555

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0404640, upper bound: 1783.0383094
time: 0.83 seconds

## Relational analysis of NS_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0406040, upper bound: 1783.0383694
time: 0.93 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -493.2639771, 1705.1289062, -491.0123291, 1697.2462158, -2190.5097656, 2196.1408691
1: -499.6026306, 1072.6654053, -497.3096619, 1067.7528076, -1567.3554688, 1569.9750977
2: -453.1403198, 1048.3687744, -451.0885925, 1043.4934082, -1496.6337891, 1499.4570312
3: -542.2065430, 1285.1954346, -539.7470703, 1279.3251953, -1821.5317383, 1824.9425049
4: -623.2268677, 1160.9147949, -620.3925171, 1155.5444336, -1778.7712402, 1781.3072510

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0406339, upper bound: 1783.0385269
time: 1.01 seconds

## Relational analysis of NS_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0406339, upper bound: 1783.0385269
time: 0.99 seconds

## BFS NS instance: NS_B1_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -402.1864014, 1399.4794922, -521.9530029, 1805.4085693, -2207.5944824, 1921.4324951
1: -408.0348816, 879.8641357, -529.7789307, 1131.8739014, -1539.9088135, 1409.6430664
2: -372.1815186, 862.3933716, -478.3341370, 1110.3736572, -1482.5551758, 1340.7275391
3: -441.2138977, 1053.3394775, -572.4196777, 1357.4888916, -1798.7027588, 1625.7591553
4: -511.9244995, 952.5472412, -659.0676880, 1231.1840820, -1743.1086426, 1611.6148682

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B1_A1_B2_A1_A1_B1

### Relational analysis result of NS_B1_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0374212, upper bound: 1783.0390830
time: 1.02 seconds

## Relational analysis of NS_B1_A1_B2_A1_A1_B2

### Relational analysis result of NS_B1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0374212, upper bound: 1783.0406339
time: 0.84 seconds

## BFS NS instance: NS_B1_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -491.0123291, 1697.2462158, -504.8154602, 1739.7373047, -2230.7495117, 2202.0615234
1: -497.3096619, 1067.7528076, -511.7325134, 1093.1666260, -1590.4763184, 1579.4853516
2: -451.0885925, 1043.4934082, -461.1958923, 1070.2629395, -1521.3515625, 1504.6893311
3: -539.7470703, 1279.3251953, -553.5442505, 1310.2337646, -1849.9808350, 1832.8693848
4: -620.3925171, 1155.5444336, -634.8883057, 1186.9227295, -1807.3151855, 1790.4327393

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B1_A1_B2_A1_A2_B1

### Relational analysis result of NS_B1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0374212, upper bound: 1783.0390830
time: 0.93 seconds

## Relational analysis of NS_B1_A1_B2_A1_A2_B2

### Relational analysis result of NS_B1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0374212, upper bound: 1783.0406339
time: 1.05 seconds

## BFS NS instance: NS_B1_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -508.0312195, 1755.5294189, -522.1917725, 1806.2250977, -2314.2563477, 2277.7211914
1: -515.6182861, 1100.9423828, -530.0278931, 1132.3687744, -1647.9869385, 1630.9702148
2: -465.3478699, 1079.5743408, -478.5468750, 1110.8658447, -1576.2137451, 1558.1212158
3: -557.2351685, 1320.5184326, -572.6833496, 1358.0944824, -1915.3294678, 1893.2017822
4: -641.0810547, 1197.5264893, -659.3652954, 1231.7442627, -1872.8253174, 1856.8918457

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B1_A1_B2_A2_A1_B1

### Relational analysis result of NS_B1_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0372359, upper bound: 1783.0372359
time: 0.75 seconds

## Relational analysis of NS_B1_A1_B2_A2_A1_B2

### Relational analysis result of NS_B1_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0372359, upper bound: 1783.0373407
time: 1.10 seconds

## BFS NS instance: NS_B1_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -597.9064941, 2049.4868164, -504.8154602, 1739.7373047, -2337.6433105, 2554.3022461
1: -605.1180420, 1290.0789795, -511.7325134, 1093.1666260, -1698.2845459, 1801.8114014
2: -543.7015381, 1260.6563721, -461.1958923, 1070.2629395, -1613.9644775, 1721.8522949
3: -656.5477905, 1547.1579590, -553.5442505, 1310.2337646, -1966.7814941, 2100.7021484
4: -748.1018066, 1399.9119873, -634.8883057, 1186.9227295, -1935.0244141, 2034.8002930

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B1_A1_B2_A2_A2_B1

### Relational analysis result of NS_B1_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0373407, upper bound: 1783.0380162
time: 0.70 seconds

## Relational analysis of NS_B1_A1_B2_A2_A2_B2

### Relational analysis result of NS_B1_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0373407, upper bound: 1783.0383466
time: 0.82 seconds

## BFS NS instance: NS_B1_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -571.6584473, 2001.3786621, -368.2507324, 1287.8657227, -1859.5240479, 2367.9577637
1: -581.4015503, 1256.5845947, -374.1566162, 808.2107544, -1388.8845215, 1630.7412109
2: -532.5600586, 1234.8784180, -341.7368469, 793.7492676, -1326.3085938, 1575.9786377
3: -629.0735474, 1502.3020020, -404.0770264, 966.9581299, -1594.1844482, 1906.3790283
4: -732.5076294, 1360.1608887, -470.4206848, 875.1803589, -1607.6879883, 1830.1601562

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_B1_B1_B1_B1

### Relational analysis result of NS_B1_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0337254, upper bound: 1783.0318170
time: 0.92 seconds

## Relational analysis of NS_B1_A2_B1_B1_B1_B2

### Relational analysis result of NS_B1_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0335833, upper bound: 1783.0318167
time: 0.81 seconds

## BFS NS instance: NS_B1_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -572.3078613, 2003.7503662, -367.4661255, 1282.3175049, -1854.6253662, 2369.5815430
1: -582.0432129, 1258.0194092, -373.3591003, 805.3966064, -1386.7293701, 1631.3785400
2: -533.1564331, 1236.2779541, -340.7346497, 790.3412476, -1323.4974365, 1576.3798828
3: -629.7321167, 1504.0302734, -403.5226746, 963.9478760, -1591.8597412, 1907.5529785
4: -733.3667603, 1361.6645508, -468.7996521, 872.3128662, -1605.6794434, 1830.0573730

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_B1_B1_B2_B1

### Relational analysis result of NS_B1_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0324219, upper bound: 1783.0317093
time: 1.19 seconds

## Relational analysis of NS_B1_A2_B1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B1_B1_B2_A1

### Relational analysis result of NS_B1_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0355668, upper bound: 1783.0305325
time: 0.72 seconds

## Relational analysis of NS_B1_A2_B1_B1_B2_A2

### Relational analysis result of NS_B1_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0355668, upper bound: 1783.0333004
time: 1.05 seconds

## BFS NS instance: NS_B1_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -556.3097534, 1953.0224609, -393.3986816, 1380.1348877, -1936.4445801, 2345.0986328
1: -565.7197266, 1223.8311768, -399.8093567, 863.8726807, -1429.1862793, 1623.6405029
2: -517.5476685, 1204.6164551, -364.8266602, 849.9347534, -1367.4824219, 1568.9899902
3: -611.9558105, 1462.9732666, -431.3116760, 1033.6906738, -1644.1298828, 1894.2849121
4: -713.1946411, 1325.4582520, -503.2226868, 936.3531494, -1649.5477295, 1828.4072266

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_B1_B2_B1_B1

### Relational analysis result of NS_B1_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0347717, upper bound: 1783.0318503
time: 0.81 seconds

## Relational analysis of NS_B1_A2_B1_B2_B1_B2

### Relational analysis result of NS_B1_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0337741, upper bound: 1783.0318498
time: 1.14 seconds

## BFS NS instance: NS_B1_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -557.4741821, 1957.1285400, -398.0320435, 1399.7076416, -1957.1818848, 2353.8891602
1: -566.8914185, 1226.3804932, -404.7124329, 874.2283936, -1440.7438965, 1631.0928955
2: -518.6187744, 1207.1047363, -368.9083557, 860.6386108, -1379.2573242, 1575.5865479
3: -613.1989746, 1466.0480957, -436.5604858, 1046.1043701, -1657.8461914, 1902.6086426
4: -714.7102661, 1328.2010498, -509.3747864, 948.0131226, -1662.7233887, 1837.3375244

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_B1_B2_B2_B1

### Relational analysis result of NS_B1_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0347450, upper bound: 1783.0318514
time: 0.92 seconds

## Relational analysis of NS_B1_A2_B1_B2_B2_B2

### Relational analysis result of NS_B1_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0337896, upper bound: 1783.0318501
time: 0.90 seconds

## BFS NS instance: NS_B1_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -563.3577881, 1973.8314209, -391.8344421, 1375.1643066, -1938.5220947, 2364.3449707
1: -572.5720215, 1239.4803467, -397.3728943, 862.7871704, -1434.6979980, 1636.8531494
2: -525.1415405, 1217.8741455, -364.7824402, 847.9744873, -1373.1154785, 1582.1210938
3: -619.7858276, 1481.8223877, -429.8407898, 1031.8033447, -1649.7404785, 1911.6630859
4: -722.5274048, 1340.5061035, -502.3764343, 932.6833496, -1655.2106934, 1842.6336670

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_A2_B2_B1_B1_A1

### Relational analysis result of NS_B1_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0334114, upper bound: 1783.0252168
time: 0.72 seconds

## Relational analysis of NS_B1_A2_B2_B1_B1_A2

### Relational analysis result of NS_B1_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0352269, upper bound: 1783.0303352
time: 0.99 seconds

## BFS NS instance: NS_B1_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -559.6238403, 1960.7557373, -474.4573059, 1653.3510742, -2212.9748535, 2433.3105469
1: -568.7949219, 1231.0135498, -481.4459229, 1034.9848633, -1602.9971924, 1712.4594727
2: -521.5722046, 1209.6031494, -436.7882080, 1017.3029785, -1538.8751221, 1645.7319336
3: -615.7000732, 1471.8264160, -520.3303833, 1239.8880615, -1853.6890869, 1991.7504883
4: -717.6650391, 1331.4803467, -602.8524170, 1123.8577881, -1841.5228271, 1933.7778320

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_B2_B1_B2_A1

### Relational analysis result of NS_B1_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0341879, upper bound: 1783.0314986
time: 0.95 seconds

## Relational analysis of NS_B1_A2_B2_B1_B2_A2

### Relational analysis result of NS_B1_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0341879, upper bound: 1783.0334879
time: 1.06 seconds

## BFS NS instance: NS_B1_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -550.0369873, 1931.9550781, -417.7021484, 1472.6682129, -2022.7050781, 2348.7145996
1: -559.0623779, 1210.8489990, -424.2553711, 919.8533936, -1478.5662842, 1635.1043701
2: -511.8376160, 1191.6704102, -387.8196411, 906.5178223, -1418.3554688, 1579.1669922
3: -605.0197144, 1447.4381104, -458.1880188, 1100.0242920, -1703.5333252, 1905.6260986
4: -705.5222168, 1310.5484619, -535.5114746, 996.2402954, -1701.7624512, 1845.9698486

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_A2_B2_B2_B1_A1

### Relational analysis result of NS_B1_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0336942, upper bound: 1783.0253250
time: 0.77 seconds

## Relational analysis of NS_B1_A2_B2_B2_B1_A2

### Relational analysis result of NS_B1_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0355090, upper bound: 1783.0304127
time: 0.94 seconds

## BFS NS instance: NS_B1_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -546.5863647, 1919.8571777, -496.8956909, 1741.2962646, -2287.8825684, 2415.1867676
1: -555.5621338, 1203.0225830, -504.5782776, 1085.4600830, -1640.5548096, 1707.6008301
2: -508.5451355, 1184.0000000, -457.2213135, 1069.2401123, -1577.7852783, 1640.7581787
3: -601.2349243, 1438.2216797, -544.8909302, 1300.3468018, -1899.9478760, 1983.0687256
4: -701.0405884, 1302.1960449, -632.6152344, 1179.8773193, -1880.9179688, 1934.4133301

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_B2_B2_B2_A1

### Relational analysis result of NS_B1_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0355099, upper bound: 1783.0338304
time: 0.99 seconds

## Relational analysis of NS_B1_A2_B2_B2_B2_A2

### Relational analysis result of NS_B1_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0355099, upper bound: 1783.0338304
time: 0.85 seconds

## BFS NS instance: NS_B2_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -378.6192017, 1316.9289551, -560.2836304, 1959.7275391, -2336.0502930, 1877.2124023
1: -383.8988342, 827.3636475, -569.6162720, 1230.4924316, -1614.3911133, 1395.7590332
2: -350.5026245, 809.8778687, -521.7007446, 1208.5479736, -1558.1212158, 1331.5786133
3: -415.4668579, 991.0310669, -616.4790649, 1471.5206299, -1886.9874268, 1605.2330322
4: -482.2958069, 895.4312134, -717.5811157, 1331.6959229, -1813.2885742, 1613.0123291

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A1_B1_A1_A1

### Relational analysis result of NS_B2_A1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0261276, upper bound: 1783.0353168
time: 0.77 seconds

## Relational analysis of NS_B2_A1_A1_B1_A1_A2

### Relational analysis result of NS_B2_A1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0259173, upper bound: 1783.0325109
time: 0.86 seconds

## BFS NS instance: NS_B2_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -427.9104614, 1492.9801025, -560.3035889, 1959.7910156, -2385.4528809, 2053.2836914
1: -434.0963745, 936.2783813, -569.6364746, 1230.5328369, -1664.6291504, 1504.7454834
2: -395.8414307, 918.7122192, -521.7174072, 1208.5887451, -1603.3479004, 1440.4296875
3: -469.2695312, 1121.6817627, -616.5024414, 1471.5695801, -1940.8391113, 1735.7215576
4: -545.1170654, 1014.5442505, -717.6043701, 1331.7432861, -1876.1239014, 1732.1486816

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A1_B1_A2_A1

### Relational analysis result of NS_B2_A1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0261276, upper bound: 1783.0360175
time: 0.78 seconds

## Relational analysis of NS_B2_A1_A1_B1_A2_A2

### Relational analysis result of NS_B2_A1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0259173, upper bound: 1783.0355848
time: 1.01 seconds

## BFS NS instance: NS_B2_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -378.6192017, 1316.9289551, -565.6705933, 1976.4404297, -2353.9538574, 1882.5993652
1: -383.8988342, 827.3636475, -574.5731812, 1240.8666992, -1624.7655029, 1401.4128418
2: -350.5026245, 809.8778687, -525.5702515, 1218.3581543, -1568.5000000, 1335.4481201
3: -415.4668579, 991.0310669, -621.8507690, 1484.5008545, -1899.9676514, 1611.3358154
4: -482.2958069, 895.4312134, -723.2058105, 1343.1975098, -1825.2790527, 1618.6369629

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A1_B2_A1_A1

### Relational analysis result of NS_B2_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0261276, upper bound: 1783.0365670
time: 2.40 seconds

## Relational analysis of NS_B2_A1_A1_B2_A1_A2

### Relational analysis result of NS_B2_A1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0259173, upper bound: 1783.0333257
time: 0.71 seconds

## BFS NS instance: NS_B2_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -427.9104614, 1492.9801025, -565.6900024, 1976.5026855, -2404.4130859, 2058.6699219
1: -434.0963745, 936.2783813, -574.5927124, 1240.9067383, -1675.0031738, 1510.8710938
2: -395.8414307, 918.7122192, -525.5864868, 1218.3977051, -1614.2391357, 1444.2987061
3: -469.2695312, 1121.6817627, -621.8734131, 1484.5485840, -1953.8181152, 1742.7910156
4: -545.1170654, 1014.5442505, -723.2285767, 1343.2430420, -1888.3601074, 1737.7728271

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A1_B2_A2_A1

### Relational analysis result of NS_B2_A1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0261276, upper bound: 1783.0381913
time: 0.89 seconds

## Relational analysis of NS_B2_A1_A1_B2_A2_A2

### Relational analysis result of NS_B2_A1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0259173, upper bound: 1783.0377851
time: 0.84 seconds

## BFS NS instance: NS_B2_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -521.0726929, 1802.6274414, -563.7461548, 1969.8394775, -2490.3222656, 2366.3735352
1: -528.9275513, 1129.9240723, -572.6299438, 1236.7335205, -1765.6611328, 1702.4443359
2: -477.6654358, 1108.4791260, -523.8265381, 1214.3173828, -1691.8782959, 1632.3055420
3: -571.5117798, 1355.2596436, -619.7510376, 1479.5104980, -2050.9606934, 1974.1218262
4: -658.1113892, 1229.1707764, -720.7966309, 1338.6898193, -1996.7148438, 1949.9674072

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_A2_B1_B2_A1

### Relational analysis result of NS_B2_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0336053, upper bound: 1783.0319089
time: 1.03 seconds

## Relational analysis of NS_B2_A1_A2_B1_B2_A2

### Relational analysis result of NS_B2_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0336053, upper bound: 1783.0322131
time: 0.79 seconds

## BFS NS instance: NS_B2_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -499.2462463, 1725.1431885, -644.7584229, 2246.1789551, -2742.0598145, 2369.9008789
1: -506.5245667, 1081.8433838, -656.0450439, 1405.9904785, -1912.5150146, 1735.3250732
2: -457.2517395, 1060.6989746, -595.6835938, 1381.8937988, -1837.7478027, 1656.3825684
3: -547.6086426, 1297.4554443, -709.0855103, 1684.4630127, -2231.0842285, 2002.9847412
4: -630.0949707, 1176.4722900, -820.6512451, 1527.9114990, -2155.9851074, 1997.1235352

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_A2_B2_B1_A1

### Relational analysis result of NS_B2_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0258800, upper bound: 1783.0301654
time: 0.85 seconds

## Relational analysis of NS_B2_A1_A2_B2_B1_A2

### Relational analysis result of NS_B2_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0258800, upper bound: 1783.0302567
time: 0.87 seconds

## BFS NS instance: NS_B2_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -521.5786133, 1804.3557129, -646.7487183, 2250.8867188, -2771.3686523, 2451.1044922
1: -529.4555054, 1130.9719238, -657.6468506, 1408.9273682, -1938.3828125, 1787.2938232
2: -478.1163330, 1109.5206299, -596.3776245, 1384.5955811, -1862.2363281, 1705.8981934
3: -572.0712280, 1356.5428467, -710.7745361, 1688.3682861, -2260.2319336, 2065.1315918
4: -658.7416382, 1230.3571777, -821.8871460, 1531.6096191, -2189.1608887, 2052.2443848

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_A2_B2_B2_A1

### Relational analysis result of NS_B2_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0316137, upper bound: 1783.0317096
time: 0.91 seconds

## Relational analysis of NS_B2_A1_A2_B2_B2_A2

### Relational analysis result of NS_B2_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0316137, upper bound: 1783.0319134
time: 0.76 seconds

## BFS NS instance: NS_B2_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -563.7890015, 1969.9785156, -574.0125732, 2006.5133057, -2569.7104492, 2543.7331543
1: -572.6735229, 1236.8220215, -583.2042236, 1259.6961670, -1831.7155762, 1819.1641846
2: -523.8625488, 1214.4053955, -533.5594482, 1237.1120605, -1760.2434082, 1747.3347168
3: -619.8012695, 1479.6166992, -631.1569214, 1506.8120117, -2124.4262695, 2108.4697266
4: -720.8470459, 1338.7919922, -734.1974487, 1363.5688477, -2084.3466797, 2072.9895020

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_A2_A1_B1_A2_B1

### Relational analysis result of NS_B2_A2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0172384, upper bound: 1783.0279657
time: 0.87 seconds

## Relational analysis of NS_B2_A2_A1_B1_A2_B2

### Relational analysis result of NS_B2_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0172384, upper bound: 1783.0355738
time: 1.02 seconds

## BFS NS instance: NS_B2_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -563.7890015, 1969.9785156, -660.6206055, 2299.9189453, -2862.6032715, 2629.6450195
1: -572.6735229, 1236.8220215, -671.8270874, 1439.7004395, -2011.6724854, 1906.6217041
2: -523.8625488, 1214.4053955, -609.4532471, 1414.9461670, -1937.7894287, 1823.1081543
3: -619.8012695, 1479.6166992, -726.0979614, 1725.1313477, -2342.6047363, 2202.1708984
4: -720.8470459, 1338.7919922, -839.8402710, 1564.8588867, -2284.6110840, 2178.3254395

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_A2_A1_B2_A2_B1

### Relational analysis result of NS_B2_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0262452, upper bound: 1783.0335983
time: 0.76 seconds

## Relational analysis of NS_B2_A2_A1_B2_A2_B2

### Relational analysis result of NS_B2_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0262452, upper bound: 1783.0349768
time: 0.85 seconds

## BFS NS instance: NS_B2_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -660.5984497, 2299.8444824, -563.7890015, 1969.9785156, -2629.6232910, 2862.5290527
1: -671.8041382, 1439.6551514, -572.6735229, 1236.8220215, -1906.5989990, 2011.6271973
2: -609.4336548, 1414.9008789, -523.8625488, 1214.4053955, -1823.0887451, 1937.7442627
3: -726.0736084, 1725.0758057, -619.8012695, 1479.6166992, -2202.1469727, 2342.5488281
4: -839.8128052, 1564.8076172, -720.8470459, 1338.7919922, -2178.2983398, 2284.5598145

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A2_A2_B1_B2_A1

### Relational analysis result of NS_B2_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0335983, upper bound: 1783.0262452
time: 0.88 seconds

## Relational analysis of NS_B2_A2_A2_B1_B2_A2

### Relational analysis result of NS_B2_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0335983, upper bound: 1783.0319971
time: 0.80 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -644.8888550, 2246.6003418, -655.7668457, 2284.6354980, -2925.8806152, 2898.5886230
1: -656.1776733, 1406.2609863, -667.2131958, 1430.0311279, -2082.9870605, 2070.3276367
2: -595.7944336, 1382.1644287, -605.7799072, 1405.6220703, -1999.1214600, 1985.6099854
3: -709.2383423, 1684.7873535, -721.1401978, 1713.2971191, -2417.0903320, 2400.5388184
4: -820.8051758, 1528.2248535, -834.5800781, 1554.1528320, -2372.7045898, 2360.5034180

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_A2_A2_B2_A1_B1

### Relational analysis result of NS_B2_A2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0256168, upper bound: 1783.0256196
time: 0.94 seconds

## Relational analysis of NS_B2_A2_A2_B2_A1_B2

### Relational analysis result of NS_B2_A2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0256169, upper bound: 1783.0260053
time: 1.06 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -646.7774048, 2250.9809570, -660.7581787, 2300.3835449, -2945.3447266, 2910.2817383
1: -657.6758423, 1408.9869385, -671.9703369, 1439.9846191, -2095.7446289, 2078.8679199
2: -596.4020386, 1384.6567383, -609.5757446, 1415.2275391, -2010.4808350, 1993.1104736
3: -710.8078003, 1688.4398193, -726.2498169, 1725.4791260, -2432.6628418, 2411.0031738
4: -821.9213867, 1531.6799316, -840.0108032, 1565.1801758, -2385.6359863, 2370.2858887

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_A2_A2_B2_A2_B1

### Relational analysis result of NS_B2_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0259057, upper bound: 1783.0302152
time: 0.81 seconds

## Relational analysis of NS_B2_A2_A2_B2_A2_B2

### Relational analysis result of NS_B2_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0259057, upper bound: 1783.0302152
time: 0.90 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.54 seconds
NS_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0390830, upper bound: 1783.0374212
NS_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0390830, upper bound: 1783.0374212
NS_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0390830, upper bound: 1783.0374212
NS_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0390830, upper bound: 1783.0374212
NS_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0404640, upper bound: 1783.0383094
NS_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0406040, upper bound: 1783.0383694
NS_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0406339, upper bound: 1783.0385269
NS_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0406339, upper bound: 1783.0385269
NS_B1_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0374212, upper bound: 1783.0390830
NS_B1_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0374212, upper bound: 1783.0406339
NS_B1_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0374212, upper bound: 1783.0390830
NS_B1_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0374212, upper bound: 1783.0406339
NS_B1_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0372359, upper bound: 1783.0372359
NS_B1_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0372359, upper bound: 1783.0373407
NS_B1_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0373407, upper bound: 1783.0380162
NS_B1_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0373407, upper bound: 1783.0383466
NS_B1_A2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0337254, upper bound: 1783.0318170
NS_B1_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0335833, upper bound: 1783.0318167
NS_B1_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0355668, upper bound: 1783.0305325
NS_B1_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0355668, upper bound: 1783.0333004
NS_B1_A2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0347717, upper bound: 1783.0318503
NS_B1_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0337741, upper bound: 1783.0318498
NS_B1_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0347450, upper bound: 1783.0318514
NS_B1_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0337896, upper bound: 1783.0318501
NS_B1_A2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0334114, upper bound: 1783.0252168
NS_B1_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0352269, upper bound: 1783.0303352
NS_B1_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0341879, upper bound: 1783.0314986
NS_B1_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0341879, upper bound: 1783.0334879
NS_B1_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0336942, upper bound: 1783.0253250
NS_B1_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0355090, upper bound: 1783.0304127
NS_B1_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0355099, upper bound: 1783.0338304
NS_B1_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0355099, upper bound: 1783.0338304
NS_B2_A1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0261276, upper bound: 1783.0353168
NS_B2_A1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0259173, upper bound: 1783.0325109
NS_B2_A1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0261276, upper bound: 1783.0360175
NS_B2_A1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0259173, upper bound: 1783.0355848
NS_B2_A1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0261276, upper bound: 1783.0365670
NS_B2_A1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0259173, upper bound: 1783.0333257
NS_B2_A1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0261276, upper bound: 1783.0381913
NS_B2_A1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0259173, upper bound: 1783.0377851
NS_B2_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0336053, upper bound: 1783.0319089
NS_B2_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0336053, upper bound: 1783.0322131
NS_B2_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0258800, upper bound: 1783.0301654
NS_B2_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0258800, upper bound: 1783.0302567
NS_B2_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0316137, upper bound: 1783.0317096
NS_B2_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0316137, upper bound: 1783.0319134
NS_B2_A2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0172384, upper bound: 1783.0279657
NS_B2_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0172384, upper bound: 1783.0355738
NS_B2_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0262452, upper bound: 1783.0335983
NS_B2_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0262452, upper bound: 1783.0349768
NS_B2_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0335983, upper bound: 1783.0262452
NS_B2_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0335983, upper bound: 1783.0319971
NS_B2_A2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0256168, upper bound: 1783.0256196
NS_B2_A2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0256169, upper bound: 1783.0260053
NS_B2_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0259057, upper bound: 1783.0302152
NS_B2_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.54
Output dim: 0, lower bound: -1783.0259057, upper bound: 1783.0302152

## BFS NS instance: NS_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -402.1864014, 1399.4794922, -402.1864014, 1399.4794922, -1801.6658936, 1801.6658936
1: -408.0348816, 879.8641357, -408.0348816, 879.8641357, -1287.8990479, 1287.8990479
2: -372.1815186, 862.3933716, -372.1815186, 862.3933716, -1234.5748291, 1234.5748291
3: -441.2138977, 1053.3394775, -441.2138977, 1053.3394775, -1494.5533447, 1494.5533447
4: -511.9244995, 952.5472412, -511.9244995, 952.5472412, -1464.4716797, 1464.4716797

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_A1_B1_A1_B1_A1_A1

### Relational analysis result of NS_B1_A1_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0367231, upper bound: 1783.0307332
time: 0.78 seconds

## Relational analysis of NS_B1_A1_B1_A1_B1_A1_A2

### Relational analysis result of NS_B1_A1_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0382970, upper bound: 1783.0331892
time: 0.83 seconds

## BFS NS instance: NS_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -507.7998657, 1754.7395020, -402.1864014, 1399.4794922, -1907.2792969, 2156.9257812
1: -515.3770752, 1100.4638672, -408.0348816, 879.8641357, -1395.2410889, 1508.4986572
2: -465.1419373, 1079.0989990, -372.1815186, 862.3933716, -1327.5351562, 1451.2805176
3: -556.9799194, 1319.9320068, -441.2138977, 1053.3394775, -1610.3193359, 1761.1458740
4: -640.7929688, 1196.9854736, -511.9244995, 952.5472412, -1593.3398438, 1708.9099121

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0357333, upper bound: 1783.0330760
time: 0.86 seconds

## Relational analysis of NS_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0382970, upper bound: 1783.0331892
time: 1.13 seconds

## BFS NS instance: NS_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -402.1864014, 1399.4794922, -491.0123291, 1697.2462158, -2099.4321289, 1890.4918213
1: -408.0348816, 879.8641357, -497.3096619, 1067.7528076, -1475.7877197, 1377.1738281
2: -372.1815186, 862.3933716, -451.0885925, 1043.4934082, -1415.6748047, 1313.4818115
3: -441.2138977, 1053.3394775, -539.7470703, 1279.3251953, -1720.5390625, 1593.0865479
4: -511.9244995, 952.5472412, -620.3925171, 1155.5444336, -1667.4689941, 1572.9394531

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_A1_B1_A1_B2_A1_A1

### Relational analysis result of NS_B1_A1_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0327759, upper bound: 1783.0304710
time: 1.03 seconds

## Relational analysis of NS_B1_A1_B1_A1_B2_A1_A2

### Relational analysis result of NS_B1_A1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0366294, upper bound: 1783.0330680
time: 0.75 seconds

## BFS NS instance: NS_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -508.0312195, 1755.5294189, -491.0123291, 1697.2462158, -2205.2773438, 2246.5417480
1: -515.6182861, 1100.9423828, -497.3096619, 1067.7528076, -1583.3710938, 1598.2520752
2: -465.3478699, 1079.5743408, -451.0885925, 1043.4934082, -1508.8413086, 1530.6628418
3: -557.2351685, 1320.5184326, -539.7470703, 1279.3251953, -1836.5601807, 1860.2655029
4: -641.0810547, 1197.5264893, -620.3925171, 1155.5444336, -1796.6254883, 1817.9189453

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B1_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_A1_B1_A1_B2_A2_A1

### Relational analysis result of NS_B1_A1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0327759, upper bound: 1783.0304710
time: 0.83 seconds

## Relational analysis of NS_B1_A1_B1_A1_B2_A2_A2

### Relational analysis result of NS_B1_A1_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0366294, upper bound: 1783.0330680
time: 0.74 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -477.4516296, 1652.4675293, -396.2396240, 1379.5590820, -1857.0106201, 2048.7070312
1: -483.6386414, 1038.8128662, -402.0656128, 867.1499634, -1350.7885742, 1440.8784180
2: -438.6103516, 1015.7286987, -366.7357178, 850.1274414, -1288.7375488, 1382.4643555
3: -524.8390503, 1244.5842285, -434.6964111, 1038.0047607, -1562.8436279, 1679.2806396
4: -603.4805908, 1124.2703857, -504.5073242, 938.7514648, -1542.2320557, 1628.7777100

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0405496, upper bound: 1783.0382921
time: 0.72 seconds

## Relational analysis of NS_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0405496, upper bound: 1783.0383094
time: 0.98 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -474.8463135, 1642.6903076, -393.4506836, 1370.8696289, -1845.7158203, 2036.1408691
1: -480.5379028, 1032.9813232, -399.2055664, 861.1635742, -1341.7014160, 1432.1868896
2: -436.0620728, 1009.9636230, -364.0051880, 844.6585083, -1280.7205811, 1373.9685059
3: -521.9380493, 1237.0174561, -431.5799255, 1030.8461914, -1552.7841797, 1668.5974121
4: -600.2056885, 1117.9093018, -500.9437866, 932.4755859, -1532.6809082, 1618.8529053

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0407138, upper bound: 1783.0383499
time: 1.07 seconds

## Relational analysis of NS_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0407138, upper bound: 1783.0383694
time: 1.12 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -491.0123291, 1697.2462158, -491.0123291, 1697.2462158, -2188.2583008, 2188.2583008
1: -497.3096619, 1067.7528076, -497.3096619, 1067.7528076, -1565.0625000, 1565.0625000
2: -451.0885925, 1043.4934082, -451.0885925, 1043.4934082, -1494.5817871, 1494.5817871
3: -539.7470703, 1279.3251953, -539.7470703, 1279.3251953, -1819.0722656, 1819.0722656
4: -620.3925171, 1155.5444336, -620.3925171, 1155.5444336, -1775.9368896, 1775.9368896

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0322579, upper bound: 1783.0204493
time: 1.03 seconds

## Relational analysis of NS_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_B1_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0211876, upper bound: 1783.0197684
time: 0.97 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -597.7236938, 2048.8688965, -491.0123291, 1697.2462158, -2294.9697266, 2539.8813477
1: -604.9286499, 1289.7016602, -497.3096619, 1067.7528076, -1672.6813965, 1787.0113525
2: -543.5396118, 1260.2811279, -451.0885925, 1043.4934082, -1587.0329590, 1711.3695068
3: -656.3476562, 1546.6984863, -539.7470703, 1279.3251953, -1935.6728516, 2086.4455566
4: -747.8767090, 1399.4870605, -620.3925171, 1155.5444336, -1903.4208984, 2019.8795166

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_A1_B1_A2_B2_A2_A1

### Relational analysis result of NS_B1_A1_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0405951, upper bound: 1783.0385269
time: 0.82 seconds

## Relational analysis of NS_B1_A1_B1_A2_B2_A2_A2

### Relational analysis result of NS_B1_A1_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0406339, upper bound: 1783.0385269
time: 0.75 seconds

## BFS NS instance: NS_B1_A1_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -402.1864014, 1399.4794922, -507.7998657, 1754.7395020, -2156.9257812, 1907.2792969
1: -408.0348816, 879.8641357, -515.3770752, 1100.4638672, -1508.4986572, 1395.2412109
2: -372.1815186, 862.3933716, -465.1419373, 1079.0989990, -1451.2805176, 1327.5351562
3: -441.2138977, 1053.3394775, -556.9799194, 1319.9320068, -1761.1458740, 1610.3193359
4: -511.9244995, 952.5472412, -640.7929688, 1196.9854736, -1708.9099121, 1593.3398438

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_A1_B2_A1_A1_B1_A1

### Relational analysis result of NS_B1_A1_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0330760, upper bound: 1783.0357333
time: 0.79 seconds

## Relational analysis of NS_B1_A1_B2_A1_A1_B1_A2

### Relational analysis result of NS_B1_A1_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0331892, upper bound: 1783.0382970
time: 0.74 seconds

## BFS NS instance: NS_B1_A1_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -402.1864014, 1399.4794922, -597.3689575, 2047.6687012, -2449.8547363, 1996.8483887
1: -408.0348816, 879.8641357, -604.5610352, 1288.9691162, -1697.0040283, 1484.4251709
2: -372.1815186, 862.3933716, -543.2254639, 1259.5535889, -1631.7349854, 1405.6188965
3: -441.2138977, 1053.3394775, -655.9591675, 1545.8055420, -1987.0194092, 1709.2985840
4: -511.9244995, 952.5472412, -747.4393311, 1398.6618652, -1910.5864258, 1699.9865723

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_A1_B2_A1_A1_B2_A1

### Relational analysis result of NS_B1_A1_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0330760, upper bound: 1783.0358871
time: 0.95 seconds

## Relational analysis of NS_B1_A1_B2_A1_A1_B2_A2

### Relational analysis result of NS_B1_A1_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0330760, upper bound: 1783.0395162
time: 1.09 seconds

## BFS NS instance: NS_B1_A1_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -491.0123291, 1697.2462158, -508.0312195, 1755.5294189, -2246.5417480, 2205.2773438
1: -497.3096619, 1067.7528076, -515.6182861, 1100.9423828, -1598.2519531, 1583.3710938
2: -451.0885925, 1043.4934082, -465.3478699, 1079.5743408, -1530.6628418, 1508.8413086
3: -539.7470703, 1279.3251953, -557.2351685, 1320.5184326, -1860.2655029, 1836.5601807
4: -620.3925171, 1155.5444336, -641.0810547, 1197.5264893, -1817.9189453, 1796.6254883

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_A1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B1_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B1_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_A1_B2_A1_A2_B1_B1

### Relational analysis result of NS_B1_A1_B2_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0286550, upper bound: 1783.0216399
time: 0.69 seconds

## Relational analysis of NS_B1_A1_B2_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B1_A1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_B2_A1_A2_B1_A1

### Relational analysis result of NS_B1_A1_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0370465, upper bound: 1783.0368781
time: 0.99 seconds

## Relational analysis of NS_B1_A1_B2_A1_A2_B1_A2

### Relational analysis result of NS_B1_A1_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0370465, upper bound: 1783.0387612
time: 0.86 seconds

## BFS NS instance: NS_B1_A1_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -491.0123291, 1697.2462158, -597.7236938, 2048.8688965, -2539.8813477, 2294.9697266
1: -497.3096619, 1067.7528076, -604.9286499, 1289.7016602, -1787.0113525, 1672.6813965
2: -451.0885925, 1043.4934082, -543.5396118, 1260.2811279, -1711.3695068, 1587.0329590
3: -539.7470703, 1279.3251953, -656.3476562, 1546.6984863, -2086.4453125, 1935.6728516
4: -620.3925171, 1155.5444336, -747.8767090, 1399.4870605, -2019.8795166, 1903.4208984

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_A1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_A1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_A1_B2_A1_A2_B2_B1

### Relational analysis result of NS_B1_A1_B2_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0286550, upper bound: 1783.0216398
time: 0.97 seconds

## Relational analysis of NS_B1_A1_B2_A1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B1_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_A1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_B2_A1_A2_B2_B1

### Relational analysis result of NS_B1_A1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0265988, upper bound: 1783.0326219
time: 1.08 seconds

## Relational analysis of NS_B1_A1_B2_A1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B1_A1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_B2_A1_A2_B2_B1

### Relational analysis result of NS_B1_A1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0348532, upper bound: 1783.0389673
time: 0.78 seconds

## Relational analysis of NS_B1_A1_B2_A1_A2_B2_B2

### Relational analysis result of NS_B1_A1_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0370465, upper bound: 1783.0391263
time: 0.93 seconds

## BFS NS instance: NS_B1_A1_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -508.0312195, 1755.5294189, -508.0312195, 1755.5294189, -2263.5605469, 2263.5605469
1: -515.6182861, 1100.9423828, -515.6182861, 1100.9423828, -1616.5604248, 1616.5604248
2: -465.3478699, 1079.5743408, -465.3478699, 1079.5743408, -1544.9222412, 1544.9222412
3: -557.2351685, 1320.5184326, -557.2351685, 1320.5184326, -1877.7532959, 1877.7532959
4: -641.0810547, 1197.5264893, -641.0810547, 1197.5264893, -1838.6075439, 1838.6075439

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_B2_A2_A1_B1_A1

### Relational analysis result of NS_B1_A1_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0347543, upper bound: 1783.0339914
time: 0.84 seconds

## Relational analysis of NS_B1_A1_B2_A2_A1_B1_A2

### Relational analysis result of NS_B1_A1_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0348284, upper bound: 1783.0348284
time: 0.86 seconds

## BFS NS instance: NS_B1_A1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -508.0312195, 1755.5294189, -597.9064941, 2049.4868164, -2557.5180664, 2353.4357910
1: -515.6182861, 1100.9423828, -605.1180420, 1290.0789795, -1805.6971436, 1706.0601807
2: -465.3478699, 1079.5743408, -543.7015381, 1260.6563721, -1726.0042725, 1623.2758789
3: -557.2351685, 1320.5184326, -656.5477905, 1547.1579590, -2104.3930664, 1977.0661621
4: -641.0810547, 1197.5264893, -748.1018066, 1399.9119873, -2040.9930420, 1945.6281738

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_A1_B2_A2_A1_B2_A1

### Relational analysis result of NS_B1_A1_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0327153, upper bound: 1783.0304601
time: 1.13 seconds

## Relational analysis of NS_B1_A1_B2_A2_A1_B2_A2

### Relational analysis result of NS_B1_A1_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0328484, upper bound: 1783.0329877
time: 0.94 seconds

## BFS NS instance: NS_B1_A1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -597.9064941, 2049.4868164, -508.0312195, 1755.5294189, -2353.4355469, 2557.5180664
1: -605.1180420, 1290.0789795, -515.6182861, 1100.9423828, -1706.0601807, 1805.6971436
2: -543.7015381, 1260.6563721, -465.3478699, 1079.5743408, -1623.2758789, 1726.0042725
3: -656.5477905, 1547.1579590, -557.2351685, 1320.5184326, -1977.0661621, 2104.3930664
4: -748.1018066, 1399.9119873, -641.0810547, 1197.5264893, -1945.6281738, 2040.9930420

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B1_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B1_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B1_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B1_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_B2_A2_A2_B1_A1

### Relational analysis result of NS_B1_A1_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0369627, upper bound: 1783.0350974
time: 0.73 seconds

## Relational analysis of NS_B1_A1_B2_A2_A2_B1_A2

### Relational analysis result of NS_B1_A1_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0369643, upper bound: 1783.0376780
time: 1.18 seconds

## BFS NS instance: NS_B1_A1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -597.9064941, 2049.4868164, -597.9064941, 2049.4868164, -2647.3933105, 2647.3933105
1: -605.1180420, 1290.0789795, -605.1180420, 1290.0789795, -1895.1968994, 1895.1968994
2: -543.7015381, 1260.6563721, -543.7015381, 1260.6563721, -1804.3577881, 1804.3577881
3: -656.5477905, 1547.1579590, -656.5477905, 1547.1579590, -2203.7058105, 2203.7058105
4: -748.1018066, 1399.9119873, -748.1018066, 1399.9119873, -2148.0136719, 2148.0136719

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B1_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_B2_A2_A2_B2_A1

### Relational analysis result of NS_B1_A1_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0369627, upper bound: 1783.0352282
time: 0.98 seconds

## Relational analysis of NS_B1_A1_B2_A2_A2_B2_A2

### Relational analysis result of NS_B1_A1_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0369643, upper bound: 1783.0376780
time: 1.17 seconds

## BFS NS instance: NS_B1_A2_B1_B1_B1_B1

### Backsubstitution after applying NS history:
0: -564.6992798, 1977.7858887, -344.1042480, 1204.3526611, -1769.0518799, 2320.1804199
1: -574.3557739, 1241.7145996, -349.7792969, 756.0926514, -1329.7392578, 1591.4938965
2: -526.2919312, 1220.4232178, -319.7716980, 742.8500366, -1269.1418457, 1539.5910645
3: -621.4290771, 1484.4189453, -377.7212830, 904.2926636, -1523.9194336, 1862.1402588
4: -723.8774414, 1344.0183105, -439.9552612, 818.7792358, -1542.6566162, 1783.5517578

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_B1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A2_B1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_A2_B1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B1_B1_B1_B1_A1

### Relational analysis result of NS_B1_A2_B1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0333857, upper bound: 1783.0290756
time: 0.91 seconds

## Relational analysis of NS_B1_A2_B1_B1_B1_B1_A2

### Relational analysis result of NS_B1_A2_B1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0333857, upper bound: 1783.0316438
time: 1.04 seconds

## BFS NS instance: NS_B1_A2_B1_B1_B1_B2

### Backsubstitution after applying NS history:
0: -539.8364868, 1887.2185059, -424.3974609, 1486.0471191, -2025.8835449, 2310.2194824
1: -549.4576416, 1186.2658691, -432.1975708, 932.5190430, -1480.8625488, 1617.9628906
2: -502.9837036, 1165.3453369, -395.3782043, 917.0604248, -1419.8078613, 1559.5919189
3: -594.9325562, 1418.1571045, -465.6274414, 1116.4405518, -1708.9289551, 1882.9720459
4: -691.0646362, 1284.0446777, -543.6010742, 1011.8004761, -1702.8651123, 1827.1776123

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B1_A2_B1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_A2_B1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_B1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A2_B1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_A2_B1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B1_B1_B1_B2_A1

### Relational analysis result of NS_B1_A2_B1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0332450, upper bound: 1783.0290698
time: 1.24 seconds

## Relational analysis of NS_B1_A2_B1_B1_B1_B2_A2

### Relational analysis result of NS_B1_A2_B1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0332458, upper bound: 1783.0316414
time: 0.87 seconds

## BFS NS instance: NS_B1_A2_B1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -565.8449707, 1981.5246582, -367.4661255, 1282.3175049, -1848.1624756, 2347.3049316
1: -575.5903320, 1243.7360840, -373.3591003, 805.3966064, -1380.2440186, 1617.0952148
2: -527.2869873, 1222.2475586, -340.7346497, 790.3412476, -1317.6281738, 1562.3133545
3: -622.7141113, 1487.1024170, -403.5226746, 963.9478760, -1584.8200684, 1890.6251221
4: -725.1313477, 1346.1383057, -468.7996521, 872.3128662, -1597.4440918, 1814.4790039

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B1_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B1_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A2_B1_B1_B2_A1_A1

### Relational analysis result of NS_B1_A2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0315464, upper bound: 1783.0276939
time: 0.97 seconds

## Relational analysis of NS_B1_A2_B1_B1_B2_A1_A2

### Relational analysis result of NS_B1_A2_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0313536, upper bound: 1783.0258157
time: 0.77 seconds

## BFS NS instance: NS_B1_A2_B1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -585.6854858, 2049.2353516, -365.6295471, 1275.8616943, -1861.5471191, 2413.3312988
1: -596.0620117, 1286.6234131, -371.5231018, 801.3157349, -1396.3061523, 1658.1462402
2: -545.1542969, 1264.3510742, -339.0630493, 786.3193970, -1331.4736328, 1602.8665771
3: -644.3798828, 1538.2939453, -401.5300903, 959.0928345, -1601.7260742, 1939.8238525
4: -749.8109741, 1393.4505615, -466.4313354, 867.8816528, -1617.6926270, 1859.5264893

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A2_B1_B1_B2_A2_A1

### Relational analysis result of NS_B1_A2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0315498, upper bound: 1783.0303851
time: 1.02 seconds

## Relational analysis of NS_B1_A2_B1_B1_B2_A2_A2

### Relational analysis result of NS_B1_A2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0313789, upper bound: 1783.0288091
time: 0.74 seconds

## BFS NS instance: NS_B1_A2_B1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -549.0645142, 1928.4960938, -372.6109924, 1309.2889404, -1858.3533936, 2299.7250977
1: -558.3658447, 1208.3826904, -378.7923279, 819.2556763, -1377.2091064, 1587.1750488
2: -511.0142212, 1189.5913086, -346.0766602, 806.4719238, -1317.4860840, 1535.2373047
3: -603.9678345, 1444.3581543, -408.6000977, 980.0673218, -1582.5429688, 1852.9581299
4: -704.2206421, 1308.6477051, -477.3728333, 888.0293579, -1592.2497559, 1785.7376709

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_A2_B1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B1_B2_B1_B1_A1

### Relational analysis result of NS_B1_A2_B1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0344909, upper bound: 1783.0291696
time: 1.05 seconds

## Relational analysis of NS_B1_A2_B1_B2_B1_B1_A2

### Relational analysis result of NS_B1_A2_B1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0344969, upper bound: 1783.0316933
time: 0.91 seconds

## BFS NS instance: NS_B1_A2_B1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -524.5629883, 1838.8559570, -435.0025330, 1529.1854248, -2053.4172363, 2272.8842773
1: -533.8204956, 1153.4378662, -443.2417908, 956.8143311, -1489.9029541, 1596.4664307
2: -487.8671875, 1135.0833740, -405.0161133, 943.2723389, -1430.9786377, 1539.2926025
3: -577.8508301, 1378.7221680, -477.2516785, 1144.9426270, -1720.8156738, 1855.5889893
4: -671.6265259, 1249.3765869, -557.5441284, 1039.3302002, -1710.9567871, 1806.6295166

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_A2_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B1_B2_B1_B2_A1

### Relational analysis result of NS_B1_A2_B1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0335301, upper bound: 1783.0291533
time: 0.88 seconds

## Relational analysis of NS_B1_A2_B1_B2_B1_B2_A2

### Relational analysis result of NS_B1_A2_B1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0335338, upper bound: 1783.0316899
time: 0.69 seconds

## BFS NS instance: NS_B1_A2_B1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -550.2224121, 1932.6005859, -376.9440002, 1328.0111084, -1878.2335205, 2308.2106934
1: -559.5343018, 1210.9267578, -383.4219971, 828.9560547, -1388.1092529, 1594.3486328
2: -512.0823364, 1192.0760498, -349.8244934, 816.5818481, -1328.6641846, 1541.4940186
3: -605.2048340, 1447.4273682, -413.5152283, 991.6746216, -1595.4494629, 1860.9423828
4: -705.7339478, 1311.3841553, -483.1311340, 899.0184326, -1604.7524414, 1794.2696533

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_A2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_A2_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B1_B2_B2_B1_A1

### Relational analysis result of NS_B1_A2_B1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0344681, upper bound: 1783.0291776
time: 1.11 seconds

## Relational analysis of NS_B1_A2_B1_B2_B2_B1_A2

### Relational analysis result of NS_B1_A2_B1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0344745, upper bound: 1783.0316982
time: 0.85 seconds

## BFS NS instance: NS_B1_A2_B1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -525.9418945, 1843.6406250, -438.1256104, 1543.2573242, -2068.1691895, 2280.8505859
1: -535.2039185, 1156.4057617, -446.8011475, 964.0130615, -1498.5070801, 1602.8280029
2: -489.1033020, 1137.9724121, -407.7462158, 951.3099976, -1440.0240479, 1544.9562988
3: -579.3173828, 1382.2951660, -480.4710083, 1153.3546143, -1730.7679443, 1862.4488525
4: -673.3723755, 1252.5694580, -561.7911987, 1047.5682373, -1720.9406738, 1814.0980225

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B1_A2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_A2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_A2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B1_B2_B2_B2_A1

### Relational analysis result of NS_B1_A2_B1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0335500, upper bound: 1783.0291560
time: 1.19 seconds

## Relational analysis of NS_B1_A2_B1_B2_B2_B2_A2

### Relational analysis result of NS_B1_A2_B1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0335538, upper bound: 1783.0316905
time: 1.04 seconds

## BFS NS instance: NS_B1_A2_B2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -529.1267090, 1853.7584229, -360.6845093, 1264.0979004, -1793.2243652, 2212.1096191
1: -537.8474121, 1164.4129639, -365.5893555, 793.4550781, -1329.9906006, 1530.0023193
2: -493.7042847, 1143.6405029, -335.8325195, 779.0247803, -1272.7288818, 1478.4964600
3: -582.2886963, 1391.8795166, -395.6876526, 949.0482178, -1528.8151855, 1787.5671387
4: -679.0774536, 1258.5726318, -462.4056396, 857.4897461, -1536.5671387, 1720.2780762

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_A2_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A2_B2_B1_B1_A1_B1

### Relational analysis result of NS_B1_A2_B2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0330761, upper bound: 1783.0251917
time: 0.86 seconds

## Relational analysis of NS_B1_A2_B2_B1_B1_A1_B2

### Relational analysis result of NS_B1_A2_B2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0330761, upper bound: 1783.0252168
time: 0.85 seconds

## BFS NS instance: NS_B1_A2_B2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -533.1325684, 1866.0130615, -383.9154053, 1348.0870361, -1881.2194824, 2249.9277344
1: -541.3665161, 1171.8532715, -389.3241882, 845.7414551, -1387.1079102, 1561.1774902
2: -496.3108215, 1150.6942139, -357.6094360, 831.2053223, -1327.5161133, 1508.3035889
3: -586.1399536, 1401.3342285, -421.1232300, 1011.3396606, -1596.5419922, 1822.4573975
4: -683.0192871, 1266.9946289, -492.4758911, 913.8974609, -1596.9167480, 1759.4704590

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_A2_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A2_B2_B1_B1_A2_B1

### Relational analysis result of NS_B1_A2_B2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0339807, upper bound: 1783.0302524
time: 0.94 seconds

## Relational analysis of NS_B1_A2_B2_B1_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0339807, upper bound: 1783.0303352
time: 1.44 seconds

## BFS NS instance: NS_B1_A2_B2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -561.3814087, 1967.0792236, -474.3594360, 1653.0187988, -2214.4001465, 2439.4750977
1: -570.5703735, 1235.2336426, -481.3437805, 1034.7830811, -1604.5539551, 1716.5773926
2: -523.3483887, 1213.7297363, -436.7012939, 1017.1023560, -1540.4506836, 1649.7327881
3: -617.6152344, 1476.7005615, -520.2214355, 1239.6409912, -1855.3352051, 1996.4714355
4: -720.0555420, 1335.8763428, -602.7307739, 1123.6292725, -1843.6846924, 1938.0324707

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B1_A2_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_A2_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B2_B1_B2_A1_B1

### Relational analysis result of NS_B1_A2_B2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0282200, upper bound: 1783.0282125
time: 0.80 seconds

## Relational analysis of NS_B1_A2_B2_B1_B2_A1_B2

### Relational analysis result of NS_B1_A2_B2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0339581, upper bound: 1783.0312060
time: 0.72 seconds

## BFS NS instance: NS_B1_A2_B2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -641.1238403, 2238.1601562, -474.9828796, 1655.1336670, -2296.2575684, 2710.8403320
1: -652.2659912, 1400.7348633, -481.9944763, 1036.0673828, -1686.4295654, 1882.7293701
2: -592.8425293, 1377.4439697, -437.2548523, 1018.3782959, -1611.2207031, 1813.8281250
3: -705.0358276, 1677.5024414, -520.9136963, 1241.2139893, -1943.2369385, 2197.8896484
4: -817.2384033, 1521.3002930, -603.5047607, 1125.0847168, -1942.3231201, 2123.3171387

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_A2_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B2_B1_B2_A2_B1

### Relational analysis result of NS_B1_A2_B2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0282200, upper bound: 1783.0293047
time: 0.95 seconds

## Relational analysis of NS_B1_A2_B2_B1_B2_A2_B2

### Relational analysis result of NS_B1_A2_B2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0282200, upper bound: 1783.0333489
time: 1.18 seconds

## BFS NS instance: NS_B1_A2_B2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -514.6170044, 1807.8903809, -392.7477417, 1386.6328125, -1901.2497559, 2198.6950684
1: -523.1122437, 1133.2360840, -399.0617371, 864.9644775, -1387.0947266, 1532.2978516
2: -479.3739624, 1114.9669189, -364.5295715, 852.9316406, -1332.3056641, 1478.7348633
3: -566.1916504, 1354.4228516, -430.7091370, 1034.1529541, -1598.1812744, 1785.1319580
4: -660.6105347, 1225.8469238, -503.7083740, 937.0169067, -1597.6273193, 1729.0363770

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_A2_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A2_B2_B2_B1_A1_B1

### Relational analysis result of NS_B1_A2_B2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0327541, upper bound: 1783.0251626
time: 1.04 seconds

## Relational analysis of NS_B1_A2_B2_B2_B1_A1_B2

### Relational analysis result of NS_B1_A2_B2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0327541, upper bound: 1783.0253250
time: 1.10 seconds

## BFS NS instance: NS_B1_A2_B2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -520.1497192, 1824.2596436, -404.2078247, 1422.4560547, -1942.6057129, 2228.4665527
1: -528.2007446, 1143.7153320, -410.4216614, 889.5754395, -1417.7761230, 1554.1368408
2: -483.4202271, 1124.6746826, -375.4869995, 875.9910889, -1359.4113770, 1500.1614990
3: -571.9058228, 1367.5664062, -443.5140076, 1064.1990967, -1635.5036621, 1811.0803223
4: -666.1430054, 1237.5100098, -518.0483398, 963.2929688, -1629.4359131, 1755.5583496

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 17

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_A2_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A2_B2_B2_B1_A2_B1

### Relational analysis result of NS_B1_A2_B2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0331921, upper bound: 1783.0302036
time: 0.76 seconds

## Relational analysis of NS_B1_A2_B2_B2_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0331921, upper bound: 1783.0304127
time: 1.14 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.29 + 416.19 = 420.49 seconds
