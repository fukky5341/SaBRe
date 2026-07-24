## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_7.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 6)
Time budget: 420 seconds
Split limit: 100
Threshold: 2861.10740463984


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-18001.9667969, 21840.2187500, -18001.9667969, 21840.2187500, -39842.1835938, 39842.1835938)
1: (-2072.5097656, 1876.1496582, -2072.5097656, 1876.1496582, -3948.6594238, 3948.6594238)
2: (-1211.9670410, 2068.2844238, -1211.9670410, 2068.2844238, -3280.2512207, 3280.2512207)
3: (-1004.3207397, 2161.7316895, -1004.3207397, 2161.7316895, -3166.0522461, 3166.0522461)
4: (-1455.2456055, 1834.0839844, -1455.2456055, 1834.0839844, -3289.3295898, 3289.3295898)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.59 + 2.20 = 4.79 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -2861.1360160, upper bound: 2861.1360160

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1275060, upper bound: 2861.1292600
time: 0.72 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1360158, upper bound: 2861.1360158
time: 0.81 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.75 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 1.75
Output dim: 3, lower bound: -2861.1275060, upper bound: 2861.1292600
NS_B2, status: Status.UNKNOWN, split count: 1, time: 1.75
Output dim: 3, lower bound: -2861.1360158, upper bound: 2861.1360158

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -17837.4277344, 21619.4960938, -17431.5410156, 21083.5625000, -38920.9921875, 39051.0390625
1: -2051.2988281, 1858.3375244, -2000.0797119, 1814.6961670, -3865.9936523, 3858.4167480
2: -1200.4310303, 2047.9550781, -1172.1784668, 1998.5010986, -3198.9313965, 3220.1335449
3: -995.0975342, 2140.2685547, -972.3207397, 2088.1425781, -3083.2399902, 3112.5891113
4: -1441.2304688, 1816.5793457, -1406.9301758, 1773.7861328, -3215.0166016, 3223.5092773

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1268551, upper bound: 2861.1264530
time: 0.83 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1269029, upper bound: 2861.1289763
time: 0.78 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -18001.9667969, 21840.2187500, -17999.4785156, 21837.5859375, -39839.5468750, 39839.6914062
1: -2072.5097656, 1876.1496582, -2072.2609863, 1875.9085693, -3948.4184570, 3948.4106445
2: -1211.9670410, 2068.2844238, -1211.8090820, 2068.0263672, -3279.9929199, 3280.0935059
3: -1004.3207397, 2161.7316895, -1004.1849365, 2161.4653320, -3165.7861328, 3165.9165039
4: -1455.2456055, 1834.0839844, -1455.0604248, 1833.8393555, -3289.0847168, 3289.1445312

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1292601, upper bound: 2861.1275060
time: 0.84 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1292601, upper bound: 2861.1275060
time: 0.79 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.20 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 4.20
Output dim: 3, lower bound: -2861.1268551, upper bound: 2861.1264530
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 4.20
Output dim: 3, lower bound: -2861.1269029, upper bound: 2861.1289763
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 4.20
Output dim: 3, lower bound: -2861.1292601, upper bound: 2861.1275060
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 4.20
Output dim: 3, lower bound: -2861.1292601, upper bound: 2861.1275060

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -17224.8320312, 20797.5937500, -17258.9296875, 20847.8574219, -38072.6875000, 38056.5234375
1: -1972.6627197, 1791.8450928, -1977.4743652, 1795.7901611, -3768.4528809, 3769.3193359
2: -1157.3686523, 1972.4677734, -1159.9354248, 1976.9487305, -3134.3173828, 3132.4025879
3: -960.6373291, 2060.4882812, -962.5965576, 2065.3046875, -3025.9418945, 3023.0847168
4: -1388.9555664, 1751.3569336, -1392.0583496, 1755.2275391, -3144.1831055, 3143.4152832

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_A1_A1

### Relational analysis result of NS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1262262, upper bound: 2861.1258524
time: 1.03 seconds

## Relational analysis of NS_B1_A1_A2

### Relational analysis result of NS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1260788, upper bound: 2861.1256195
time: 1.01 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -17834.5781250, 21616.3281250, -17431.5410156, 21083.5625000, -38918.1406250, 39047.8671875
1: -2050.9868164, 1858.0574951, -2000.0797119, 1814.6961670, -3865.6826172, 3858.1372070
2: -1200.2474365, 2047.6503906, -1172.1784668, 1998.5010986, -3198.7475586, 3219.8288574
3: -994.9448853, 2139.9501953, -972.3207397, 2088.1425781, -3083.0871582, 3112.2709961
4: -1441.0144043, 1816.2961426, -1406.9301758, 1773.7861328, -3214.8005371, 3223.2258301

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1268976, upper bound: 2861.1289697
time: 0.94 seconds

## Relational analysis of NS_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1268976, upper bound: 2861.1289763
time: 0.91 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -17431.5410156, 21083.5625000, -17999.4785156, 21837.5859375, -39269.1210938, 39083.0390625
1: -2000.0797119, 1814.6961670, -2072.2609863, 1875.9085693, -3875.9880371, 3886.9570312
2: -1172.1784668, 1998.5010986, -1211.8090820, 2068.0263672, -3240.2048340, 3210.3098145
3: -972.3207397, 2088.1425781, -1004.1849365, 2161.4653320, -3133.7861328, 3092.3271484
4: -1406.9301758, 1773.7861328, -1455.0604248, 1833.8393555, -3240.7692871, 3228.8466797

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_B1

### Relational analysis result of NS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1264530, upper bound: 2861.1268550
time: 0.74 seconds

## Relational analysis of NS_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1269029, upper bound: 2861.1269029
time: 0.82 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -17999.4785156, 21837.5859375, -17999.4785156, 21837.5859375, -39837.0546875, 39837.0546875
1: -2072.2609863, 1875.9085693, -2072.2609863, 1875.9085693, -3948.1694336, 3948.1694336
2: -1211.8090820, 2068.0263672, -1211.8090820, 2068.0263672, -3279.8352051, 3279.8352051
3: -1004.1849365, 2161.4653320, -1004.1849365, 2161.4653320, -3165.6501465, 3165.6501465
4: -1455.0604248, 1833.8393555, -1455.0604248, 1833.8393555, -3288.8994141, 3288.8996582

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_A1

### Relational analysis result of NS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1268551, upper bound: 2861.1264530
time: 0.80 seconds

## Relational analysis of NS_B2_A2_A2

### Relational analysis result of NS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1269029, upper bound: 2861.1269029
time: 0.95 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.35 seconds
NS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 3, lower bound: -2861.1262262, upper bound: 2861.1258524
NS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 3, lower bound: -2861.1260788, upper bound: 2861.1256195
NS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 3, lower bound: -2861.1268976, upper bound: 2861.1289697
NS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 3, lower bound: -2861.1268976, upper bound: 2861.1289763
NS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 3, lower bound: -2861.1264530, upper bound: 2861.1268550
NS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 3, lower bound: -2861.1269029, upper bound: 2861.1269029
NS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 3, lower bound: -2861.1268551, upper bound: 2861.1264530
NS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 3, lower bound: -2861.1269029, upper bound: 2861.1269029

## BFS NS instance: NS_B1_A1_A1

### Backsubstitution after applying NS history:
0: -16885.3085938, 20402.5859375, -17035.8222656, 20587.9628906, -37473.2734375, 37438.4023438
1: -1935.4669189, 1757.3171387, -1953.0092773, 1773.0833740, -3708.5502930, 3710.3264160
2: -1135.2736816, 1934.3276367, -1145.3984375, 1951.8609619, -3087.1347656, 3079.7260742
3: -941.7879639, 2021.4536133, -950.2089844, 2039.6330566, -2981.4206543, 2971.6625977
4: -1362.1832275, 1717.5556641, -1374.4458008, 1732.9979248, -3095.1801758, 3092.0014648

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_A1_A1_B1

### Relational analysis result of NS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259914, upper bound: 2861.1256195
time: 0.75 seconds

## Relational analysis of NS_B1_A1_A1_B2

### Relational analysis result of NS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259914, upper bound: 2861.1256195
time: 1.14 seconds

## BFS NS instance: NS_B1_A1_A2

### Backsubstitution after applying NS history:
0: -17449.0214844, 21000.9023438, -17192.3066406, 20773.1699219, -38222.1914062, 38193.2031250
1: -1989.5050049, 1812.2502441, -1970.5407715, 1788.9957275, -3778.5007324, 3782.7910156
2: -1170.7468262, 1993.6561279, -1155.5952148, 1969.7204590, -3140.4672852, 3149.2512207
3: -973.4267578, 2080.5781250, -958.9004517, 2057.8046875, -3031.2312012, 3039.4785156
4: -1405.3482666, 1771.4483643, -1386.8446045, 1748.7387695, -3154.0866699, 3158.2929688

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 41

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_A1_A2_B1

### Relational analysis result of NS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259914, upper bound: 2861.1256195
time: 0.78 seconds

## Relational analysis of NS_B1_A1_A2_B2

### Relational analysis result of NS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259914, upper bound: 2861.1256195
time: 0.86 seconds

## BFS NS instance: NS_B1_A2_B1

### Backsubstitution after applying NS history:
0: -17834.5781250, 21616.3281250, -16978.2148438, 20441.0957031, -38275.6679688, 38594.5429688
1: -2050.9868164, 1858.0574951, -1938.6414795, 1764.0927734, -3815.0795898, 3796.6989746
2: -1200.2474365, 2047.6503906, -1139.2817383, 1941.0139160, -3141.2602539, 3186.9316406
3: -994.9448853, 2139.9501953, -946.4141846, 2026.6416016, -3021.5864258, 3086.3640137
4: -1441.0144043, 1816.2961426, -1367.0740967, 1723.9722900, -3164.9858398, 3183.3701172

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_A2_B1_B1

### Relational analysis result of NS_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1260184, upper bound: 2861.1288376
time: 1.36 seconds

## Relational analysis of NS_B1_A2_B1_B2

### Relational analysis result of NS_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1266629, upper bound: 2861.1288935
time: 0.96 seconds

## BFS NS instance: NS_B1_A2_B2

### Backsubstitution after applying NS history:
0: -17834.5781250, 21616.3281250, -17426.7519531, 21078.4082031, -38912.9804688, 39043.0781250
1: -2050.9868164, 1858.0574951, -1999.5897217, 1814.2269287, -3865.2138672, 3857.6472168
2: -1200.2474365, 2047.6503906, -1171.8747559, 1998.0010986, -3198.2482910, 3219.5251465
3: -994.9448853, 2139.9501953, -972.0583496, 2087.6232910, -3082.5678711, 3112.0080566
4: -1441.0144043, 1816.2961426, -1406.5701904, 1773.3284912, -3214.3425293, 3222.8662109

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_A2_B2_B1

### Relational analysis result of NS_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1260184, upper bound: 2861.1288442
time: 0.78 seconds

## Relational analysis of NS_B1_A2_B2_B2

### Relational analysis result of NS_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1266629, upper bound: 2861.1289012
time: 0.84 seconds

## BFS NS instance: NS_B2_A1_B1

### Backsubstitution after applying NS history:
0: -17258.9296875, 20847.8574219, -17376.6796875, 21005.1699219, -38264.1015625, 38224.5390625
1: -1977.4743652, 1795.7901611, -1992.5794678, 1808.4730225, -3785.9472656, 3788.3696289
2: -1159.9354248, 1976.9487305, -1168.1293945, 1991.4453125, -3151.3806152, 3145.0781250
3: -962.5965576, 2065.3046875, -969.1884766, 2080.5837402, -3043.1801758, 3034.4931641
4: -1392.0583496, 1755.2275391, -1402.0469971, 1767.6793213, -3159.7377930, 3157.2744141

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B2_A1_B1_A1

### Relational analysis result of NS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1260019
time: 0.71 seconds

## Relational analysis of NS_B2_A1_B1_A2

### Relational analysis result of NS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1260788
time: 0.76 seconds

## BFS NS instance: NS_B2_A1_B2

### Backsubstitution after applying NS history:
0: -17431.5410156, 21083.5625000, -17996.8652344, 21834.6718750, -39266.2070312, 39080.4140625
1: -2000.0797119, 1814.6961670, -2071.9863281, 1875.6477051, -3875.7272949, 3886.6821289
2: -1172.1784668, 1998.5010986, -1211.6391602, 2067.7456055, -3239.9240723, 3210.1398926
3: -972.3207397, 2088.1425781, -1004.0410156, 2161.1728516, -3133.4936523, 3092.1831055
4: -1406.9301758, 1773.7861328, -1454.8593750, 1833.5747070, -3240.5048828, 3228.6455078

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B2_A1_B2_A1

### Relational analysis result of NS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1288442, upper bound: 2861.1260267
time: 0.83 seconds

## Relational analysis of NS_B2_A1_B2_A2

### Relational analysis result of NS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1289011, upper bound: 2861.1266729
time: 0.77 seconds

## BFS NS instance: NS_B2_A2_A1

### Backsubstitution after applying NS history:
0: -17376.6796875, 21005.1699219, -17819.8574219, 21594.5800781, -38971.2578125, 38825.0273438
1: -1992.5794678, 1808.4730225, -2048.9045410, 1856.3593750, -3848.9389648, 3857.3774414
2: -1168.1293945, 1991.4453125, -1199.1433105, 2045.7071533, -3213.8364258, 3190.5886230
3: -969.1884766, 2080.5837402, -994.0997925, 2137.8608398, -3107.0493164, 3074.6835938
4: -1402.0469971, 1767.6793213, -1439.6812744, 1814.6199951, -3216.6669922, 3207.3605957

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_A1_A1

### Relational analysis result of NS_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1262262, upper bound: 2861.1258524
time: 0.77 seconds

## Relational analysis of NS_B2_A2_A1_A2

### Relational analysis result of NS_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1260788, upper bound: 2861.1256195
time: 0.73 seconds

## BFS NS instance: NS_B2_A2_A2

### Backsubstitution after applying NS history:
0: -17996.8652344, 21834.6718750, -17999.4785156, 21837.5859375, -39834.4296875, 39834.1406250
1: -2071.9863281, 1875.6477051, -2072.2609863, 1875.9085693, -3947.8950195, 3947.9086914
2: -1211.6391602, 2067.7456055, -1211.8090820, 2068.0263672, -3279.6652832, 3279.5546875
3: -1004.0410156, 2161.1728516, -1004.1849365, 2161.4653320, -3165.5063477, 3165.3579102
4: -1454.8593750, 1833.5747070, -1455.0604248, 1833.8393555, -3288.6982422, 3288.6347656

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_A2_B1

### Relational analysis result of NS_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1264530, upper bound: 2861.1289239
time: 0.79 seconds

## Relational analysis of NS_B2_A2_A2_B2

### Relational analysis result of NS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1264530, upper bound: 2861.1360154
time: 0.77 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.20 seconds
NS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.20
Output dim: 3, lower bound: -2861.1259914, upper bound: 2861.1256195
NS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.20
Output dim: 3, lower bound: -2861.1259914, upper bound: 2861.1256195
NS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.20
Output dim: 3, lower bound: -2861.1259914, upper bound: 2861.1256195
NS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.20
Output dim: 3, lower bound: -2861.1259914, upper bound: 2861.1256195
NS_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.20
Output dim: 3, lower bound: -2861.1260184, upper bound: 2861.1288376
NS_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.20
Output dim: 3, lower bound: -2861.1266629, upper bound: 2861.1288935
NS_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.20
Output dim: 3, lower bound: -2861.1260184, upper bound: 2861.1288442
NS_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.20
Output dim: 3, lower bound: -2861.1266629, upper bound: 2861.1289012
NS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.20
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1260019
NS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.20
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1260788
NS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.20
Output dim: 3, lower bound: -2861.1288442, upper bound: 2861.1260267
NS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.20
Output dim: 3, lower bound: -2861.1289011, upper bound: 2861.1266729
NS_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 4.20
Output dim: 3, lower bound: -2861.1262262, upper bound: 2861.1258524
NS_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.20
Output dim: 3, lower bound: -2861.1260788, upper bound: 2861.1256195
NS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.20
Output dim: 3, lower bound: -2861.1264530, upper bound: 2861.1289239
NS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.20
Output dim: 3, lower bound: -2861.1264530, upper bound: 2861.1360154

## BFS NS instance: NS_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -16885.3085938, 20402.5859375, -16919.5332031, 20453.0253906, -37338.3320312, 37322.1171875
1: -1935.4669189, 1757.3171387, -1940.2930908, 1761.2772217, -3696.7441406, 3697.6103516
2: -1135.2736816, 1934.3276367, -1137.8489990, 1938.8217773, -3074.0954590, 3072.1765137
3: -941.7879639, 2021.4536133, -943.7532349, 2026.2850342, -2968.0729980, 2965.2065430
4: -1362.1832275, 1717.5556641, -1365.2961426, 1721.4396973, -3083.6230469, 3082.8518066

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_A1_A1_B1_A1

### Relational analysis result of NS_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1260138, upper bound: 2861.1258524
time: 0.80 seconds

## Relational analysis of NS_B1_A1_A1_B1_A2

### Relational analysis result of NS_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1260138, upper bound: 2861.1258524
time: 0.99 seconds

## BFS NS instance: NS_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -16885.3085938, 20402.5859375, -17483.0937500, 21051.1035156, -37936.4101562, 37885.6796875
1: -1935.4669189, 1757.3171387, -1994.3098145, 1816.1942139, -3751.6608887, 3751.6269531
2: -1135.2736816, 1934.3276367, -1173.3135986, 1998.1241455, -3133.3979492, 3107.6411133
3: -941.7879639, 2021.4536133, -975.3895874, 2085.3896484, -3027.1772461, 2996.8432617
4: -1362.1832275, 1717.5556641, -1408.4492188, 1775.3115234, -3137.4943848, 3126.0048828

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_A1_A1_B2_A1

### Relational analysis result of NS_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1260138, upper bound: 2861.1258524
time: 0.75 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2

### Relational analysis result of NS_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1260138, upper bound: 2861.1258524
time: 0.82 seconds

## BFS NS instance: NS_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -17449.0214844, 21000.9023438, -16919.5332031, 20453.0253906, -37902.0468750, 37920.4296875
1: -1989.5050049, 1812.2502441, -1940.2930908, 1761.2772217, -3750.7822266, 3752.5434570
2: -1170.7468262, 1993.6561279, -1137.8489990, 1938.8217773, -3109.5686035, 3131.5046387
3: -973.4267578, 2080.5781250, -943.7532349, 2026.2850342, -2999.7114258, 3024.3308105
4: -1405.3482666, 1771.4483643, -1365.2961426, 1721.4396973, -3126.7875977, 3136.7446289

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B1_A1_A2_B1_B1

### Relational analysis result of NS_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259890, upper bound: 2861.1256195
time: 0.91 seconds

## Relational analysis of NS_B1_A1_A2_B1_B2

### Relational analysis result of NS_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259890, upper bound: 2861.1256195
time: 0.85 seconds

## BFS NS instance: NS_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -17449.0214844, 21000.9023438, -17483.0937500, 21051.1035156, -38500.1250000, 38483.9960938
1: -1989.5050049, 1812.2502441, -1994.3098145, 1816.1942139, -3805.6989746, 3806.5595703
2: -1170.7468262, 1993.6561279, -1173.3135986, 1998.1241455, -3168.8710938, 3166.9697266
3: -973.4267578, 2080.5781250, -975.3895874, 2085.3896484, -3058.8156738, 3055.9677734
4: -1405.3482666, 1771.4483643, -1408.4492188, 1775.3115234, -3180.6589355, 3179.8974609

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_A1_A2_B2_A1

### Relational analysis result of NS_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259914, upper bound: 2861.1256196
time: 0.93 seconds

## Relational analysis of NS_B1_A1_A2_B2_A2

### Relational analysis result of NS_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259914, upper bound: 2861.1256195
time: 0.84 seconds

## BFS NS instance: NS_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -17610.7148438, 21355.4687500, -16638.1113281, 20045.7187500, -37656.4335938, 37993.5742188
1: -2026.4188232, 1835.2718506, -1901.4204102, 1729.5068359, -3755.9257812, 3736.6923828
2: -1185.6633301, 2022.4584961, -1117.1499023, 1902.8555908, -3088.5190430, 3139.6083984
3: -982.5181885, 2114.1757812, -927.5325317, 1987.5738525, -2970.0920410, 3041.7082520
4: -1423.3345947, 1793.9914551, -1340.2596436, 1690.1424561, -3113.4763184, 3134.2509766

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_A2_B1_B1_A1

### Relational analysis result of NS_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1260184, upper bound: 2861.1285553
time: 0.78 seconds

## Relational analysis of NS_B1_A2_B1_B1_A2

### Relational analysis result of NS_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1260184, upper bound: 2861.1288376
time: 0.78 seconds

## BFS NS instance: NS_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -17769.3378906, 21543.3515625, -17198.4296875, 20641.5976562, -38410.9375000, 38741.7773438
1: -2044.2108154, 1851.4067383, -1955.2071533, 1784.1507568, -3828.3615723, 3806.6137695
2: -1196.0000000, 2040.5895996, -1152.4566650, 1962.0009766, -3158.0007324, 3193.0463867
3: -991.3273315, 2132.6125488, -958.9920654, 2046.4283447, -3037.7551270, 3091.6044922
4: -1435.9119873, 1809.9600830, -1383.2012939, 1743.8574219, -3179.7692871, 3193.1611328

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_A2_B1_B2_A1

### Relational analysis result of NS_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1264598, upper bound: 2861.1285553
time: 0.76 seconds

## Relational analysis of NS_B1_A2_B1_B2_A2

### Relational analysis result of NS_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1264598, upper bound: 2861.1288935
time: 0.81 seconds

## BFS NS instance: NS_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -17610.7148438, 21355.4687500, -17086.6757812, 20682.7519531, -38293.4687500, 38442.1406250
1: -2026.4188232, 1835.2718506, -1962.3250732, 1779.6385498, -3806.0573730, 3797.5969238
2: -1185.6633301, 2022.4584961, -1149.7426758, 1959.7862549, -3145.4489746, 3172.2011719
3: -982.5181885, 2114.1757812, -953.1752319, 2048.5190430, -3031.0371094, 3067.3508301
4: -1423.3345947, 1793.9914551, -1379.7482910, 1739.4660645, -3162.8007812, 3173.7395020

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_A2_B2_B1_A1

### Relational analysis result of NS_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1260267, upper bound: 2861.1285621
time: 1.13 seconds

## Relational analysis of NS_B1_A2_B2_B1_A2

### Relational analysis result of NS_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1260267, upper bound: 2861.1288443
time: 0.82 seconds

## BFS NS instance: NS_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -17769.3378906, 21543.3515625, -17652.0742188, 21282.1328125, -39051.4687500, 39195.4257812
1: -2044.2108154, 1851.4067383, -2016.4621582, 1834.7119141, -3878.9228516, 3867.8688965
2: -1196.0000000, 2040.5895996, -1185.3020020, 2019.2042236, -3215.2041016, 3225.8916016
3: -991.3273315, 2132.6125488, -984.9135742, 2107.7641602, -3099.0910645, 3117.5261230
4: -1435.9119873, 1809.9600830, -1423.0430908, 1793.4512939, -3229.3627930, 3233.0021973

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_A2_B2_B2_A1

### Relational analysis result of NS_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1264685, upper bound: 2861.1285623
time: 0.88 seconds

## Relational analysis of NS_B1_A2_B2_B2_A2

### Relational analysis result of NS_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1264685, upper bound: 2861.1289012
time: 0.99 seconds

## BFS NS instance: NS_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -16919.5332031, 20453.0253906, -17153.1269531, 20744.7636719, -37664.2890625, 37606.1484375
1: -1940.2930908, 1761.2772217, -1968.0589600, 1785.7193604, -3726.0119629, 3729.3361816
2: -1137.8489990, 1938.8217773, -1153.5637207, 1966.3010254, -3104.1499023, 3092.3854980
3: -943.7532349, 2026.2850342, -956.7766113, 2054.8591309, -2998.6115723, 2983.0615234
4: -1365.2961426, 1721.4396973, -1384.3968506, 1745.4039307, -3110.6997070, 3105.8364258

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A1_B1_A1_B1

### Relational analysis result of NS_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1259914
time: 1.00 seconds

## Relational analysis of NS_B2_A1_B1_A1_B2

### Relational analysis result of NS_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1259914
time: 0.78 seconds

## BFS NS instance: NS_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -17483.0937500, 21051.1035156, -17310.4589844, 20931.0156250, -38414.1093750, 38361.5625000
1: -1994.3098145, 1816.1942139, -1985.6939697, 1801.7210693, -3796.0307617, 3801.8879395
2: -1173.3135986, 1998.1241455, -1163.8173828, 1984.2670898, -3157.5805664, 3161.9414062
3: -975.3895874, 2085.3896484, -965.5156860, 2073.1354980, -3048.5249023, 3050.9047852
4: -1408.4492188, 1775.3115234, -1396.8666992, 1761.2319336, -3169.6811523, 3172.1779785

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 41

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A1_B1_A2_B1

### Relational analysis result of NS_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1260788
time: 0.79 seconds

## Relational analysis of NS_B2_A1_B1_A2_B2

### Relational analysis result of NS_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1260788
time: 0.87 seconds

## BFS NS instance: NS_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -17091.5937500, 20688.0625000, -17772.5976562, 21573.4316406, -38665.0234375, 38460.6562500
1: -1962.8297119, 1780.1209717, -2047.3842773, 1852.8277588, -3815.6574707, 3827.5053711
2: -1150.0550537, 1960.3010254, -1197.0322266, 2042.5119629, -3192.5666504, 3157.3327637
3: -953.4449463, 2049.0534668, -991.5979614, 2135.3588867, -3088.8034668, 3040.6511230
4: -1380.1186523, 1739.9373779, -1437.1496582, 1811.2342529, -3191.3530273, 3177.0869141

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A1_B2_A1_B1

### Relational analysis result of NS_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1285623, upper bound: 2861.1260267
time: 0.81 seconds

## Relational analysis of NS_B2_A1_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1285623, upper bound: 2861.1260267
time: 0.85 seconds

## BFS NS instance: NS_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -17656.8007812, 21287.2441406, -17931.8457031, 21761.9804688, -39418.7734375, 39219.0781250
1: -2016.9482422, 1835.1765137, -2065.2404785, 1869.0228271, -3885.9694824, 3900.4169922
2: -1185.6025391, 2019.6976318, -1207.4073486, 2060.7116699, -3246.3142090, 3227.1042480
3: -985.1722412, 2108.2783203, -1000.4395142, 2153.8625488, -3139.0346680, 3108.7172852
4: -1423.3986816, 1793.9025879, -1449.7738037, 1827.2625732, -3250.6611328, 3243.6760254

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A1_B2_A2_B1

### Relational analysis result of NS_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1285623, upper bound: 2861.1264685
time: 0.99 seconds

## Relational analysis of NS_B2_A1_B2_A2_B2

### Relational analysis result of NS_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1285623, upper bound: 2861.1266729
time: 1.18 seconds

## BFS NS instance: NS_B2_A2_A1_A1

### Backsubstitution after applying NS history:
0: -17036.6074219, 20609.5234375, -17595.9824219, 21333.6992188, -38370.3007812, 38205.4960938
1: -1955.3148193, 1773.8862305, -2024.3354492, 1833.5718994, -3788.8859863, 3798.2216797
2: -1145.9982910, 1953.2342529, -1184.5577393, 2020.5129395, -3166.5112305, 3137.7919922
3: -950.3074951, 2041.4816895, -981.6718140, 2112.0852051, -3062.3925781, 3023.1535645
4: -1375.2265625, 1733.8201904, -1421.9998779, 1792.3151855, -3167.5412598, 3155.8198242

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_A1_A1_B1

### Relational analysis result of NS_B2_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259914, upper bound: 2861.1256195
time: 0.90 seconds

## Relational analysis of NS_B2_A2_A1_A1_B2

### Relational analysis result of NS_B2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259914, upper bound: 2861.1256195
time: 0.75 seconds

## BFS NS instance: NS_B2_A2_A1_A2

### Backsubstitution after applying NS history:
0: -17602.0312500, 21209.1093750, -17754.6269531, 21521.5859375, -39123.6171875, 38963.7343750
1: -2009.4707031, 1828.9714355, -2042.1279297, 1849.7081299, -3859.1787109, 3871.0991211
2: -1181.5650635, 2012.6750488, -1194.8959961, 2038.6441650, -3220.2092285, 3207.5708008
3: -982.0476685, 2100.7414551, -990.4821777, 2130.5222168, -3112.5698242, 3091.2236328
4: -1418.5305176, 1787.8250732, -1434.5783691, 1808.2838135, -3226.8139648, 3222.4033203

Time for backsubstitution: 2.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_A1_A2_B1

### Relational analysis result of NS_B2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259914, upper bound: 2861.1256195
time: 1.08 seconds

## Relational analysis of NS_B2_A2_A1_A2_B2

### Relational analysis result of NS_B2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259914, upper bound: 2861.1256196
time: 0.82 seconds

## BFS NS instance: NS_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -17996.8652344, 21834.6718750, -17376.6796875, 21005.1699219, -39002.0195312, 39211.3437500
1: -2071.9863281, 1875.6477051, -1992.5794678, 1808.4730225, -3880.4594727, 3868.2270508
2: -1211.6391602, 2067.7456055, -1168.1293945, 1991.4453125, -3203.0844727, 3235.8750000
3: -1004.0410156, 2161.1728516, -969.1884766, 2080.5837402, -3084.6247559, 3130.3613281
4: -1454.8593750, 1833.5747070, -1402.0469971, 1767.6793213, -3222.5383301, 3235.6215820

Time for backsubstitution: 2.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_A2_B1_B1

### Relational analysis result of NS_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1258524, upper bound: 2861.1286767
time: 0.75 seconds

## Relational analysis of NS_B2_A2_A2_B1_B2

### Relational analysis result of NS_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1286453
time: 0.80 seconds

## BFS NS instance: NS_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -17996.8652344, 21834.6718750, -17996.8652344, 21834.6718750, -39831.5156250, 39831.5156250
1: -2071.9863281, 1875.6477051, -2071.9863281, 1875.6477051, -3947.6340332, 3947.6340332
2: -1211.6391602, 2067.7456055, -1211.6391602, 2067.7456055, -3279.3847656, 3279.3847656
3: -1004.0410156, 2161.1728516, -1004.0410156, 2161.1728516, -3165.2138672, 3165.2138672
4: -1454.8593750, 1833.5747070, -1454.8593750, 1833.5747070, -3288.4333496, 3288.4333496

Time for backsubstitution: 2.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_A2_B2_B1

### Relational analysis result of NS_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1258524, upper bound: 2861.1358747
time: 1.04 seconds

## Relational analysis of NS_B2_A2_A2_B2_B2

### Relational analysis result of NS_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1358747
time: 0.84 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.59 seconds
NS_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -2861.1260138, upper bound: 2861.1258524
NS_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -2861.1260138, upper bound: 2861.1258524
NS_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -2861.1260138, upper bound: 2861.1258524
NS_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -2861.1260138, upper bound: 2861.1258524
NS_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -2861.1259890, upper bound: 2861.1256195
NS_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -2861.1259890, upper bound: 2861.1256195
NS_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -2861.1259914, upper bound: 2861.1256196
NS_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -2861.1259914, upper bound: 2861.1256195
NS_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -2861.1260184, upper bound: 2861.1285553
NS_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -2861.1260184, upper bound: 2861.1288376
NS_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -2861.1264598, upper bound: 2861.1285553
NS_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -2861.1264598, upper bound: 2861.1288935
NS_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -2861.1260267, upper bound: 2861.1285621
NS_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -2861.1260267, upper bound: 2861.1288443
NS_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -2861.1264685, upper bound: 2861.1285623
NS_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -2861.1264685, upper bound: 2861.1289012
NS_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1259914
NS_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1259914
NS_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1260788
NS_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1260788
NS_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -2861.1285623, upper bound: 2861.1260267
NS_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -2861.1285623, upper bound: 2861.1260267
NS_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -2861.1285623, upper bound: 2861.1264685
NS_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -2861.1285623, upper bound: 2861.1266729
NS_B2_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -2861.1259914, upper bound: 2861.1256195
NS_B2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -2861.1259914, upper bound: 2861.1256195
NS_B2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -2861.1259914, upper bound: 2861.1256195
NS_B2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -2861.1259914, upper bound: 2861.1256196
NS_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -2861.1258524, upper bound: 2861.1286767
NS_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1286453
NS_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -2861.1258524, upper bound: 2861.1358747
NS_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.59
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1358747

## BFS NS instance: NS_B1_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -16638.1113281, 20045.7187500, -16919.5332031, 20453.0253906, -37091.1328125, 36965.2500000
1: -1901.4204102, 1729.5068359, -1940.2930908, 1761.2772217, -3662.6977539, 3669.7998047
2: -1117.1499023, 1902.8555908, -1137.8489990, 1938.8217773, -3055.9716797, 3040.7043457
3: -927.5325317, 1987.5738525, -943.7532349, 2026.2850342, -2953.8176270, 2931.3266602
4: -1340.2596436, 1690.1424561, -1365.2961426, 1721.4396973, -3061.6992188, 3055.4384766

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B1_A1_A1_B1_A1_B1

### Relational analysis result of NS_B1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1260087, upper bound: 2861.1258524
time: 0.81 seconds

## Relational analysis of NS_B1_A1_A1_B1_A1_B2

### Relational analysis result of NS_B1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1260087, upper bound: 2861.1258524
time: 0.87 seconds

## BFS NS instance: NS_B1_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -17036.6074219, 20609.5234375, -16919.5332031, 20453.0253906, -37489.6289062, 37529.0468750
1: -1955.3148193, 1773.8862305, -1940.2930908, 1761.2772217, -3716.5920410, 3714.1791992
2: -1145.9982910, 1953.2342529, -1137.8489990, 1938.8217773, -3084.8200684, 3091.0827637
3: -950.3074951, 2041.4816895, -943.7532349, 2026.2850342, -2976.5925293, 2985.2343750
4: -1375.2265625, 1733.8201904, -1365.2961426, 1721.4396973, -3096.6662598, 3099.1162109

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A1_B1_A2_A1

### Relational analysis result of NS_B1_A1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1257322, upper bound: 2861.1258316
time: 0.80 seconds

## Relational analysis of NS_B1_A1_A1_B1_A2_A2

### Relational analysis result of NS_B1_A1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1257303, upper bound: 2861.1251497
time: 0.73 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -16638.1113281, 20045.7187500, -17483.0937500, 21051.1035156, -37689.2109375, 37528.8125000
1: -1901.4204102, 1729.5068359, -1994.3098145, 1816.1942139, -3717.6142578, 3723.8161621
2: -1117.1499023, 1902.8555908, -1173.3135986, 1998.1241455, -3115.2739258, 3076.1691895
3: -927.5325317, 1987.5738525, -975.3895874, 2085.3896484, -3012.9218750, 2962.9633789
4: -1340.2596436, 1690.1424561, -1408.4492188, 1775.3115234, -3115.5710449, 3098.5913086

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B1_A1_A1_B2_A1_B1

### Relational analysis result of NS_B1_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1262219, upper bound: 2861.1258524
time: 0.82 seconds

## Relational analysis of NS_B1_A1_A1_B2_A1_B2

### Relational analysis result of NS_B1_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1262219, upper bound: 2861.1258524
time: 0.77 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -17036.6074219, 20609.5234375, -17483.0937500, 21051.1035156, -38087.7070312, 38092.6093750
1: -1955.3148193, 1773.8862305, -1994.3098145, 1816.1942139, -3771.5085449, 3768.1960449
2: -1145.9982910, 1953.2342529, -1173.3135986, 1998.1241455, -3144.1225586, 3126.5478516
3: -950.3074951, 2041.4816895, -975.3895874, 2085.3896484, -3035.6970215, 3016.8710938
4: -1375.2265625, 1733.8201904, -1408.4492188, 1775.3115234, -3150.5380859, 3142.2692871

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_B1_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B1_A1_A1_B2_A2_A1

### Relational analysis result of NS_B1_A1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259991, upper bound: 2861.1253033
time: 0.78 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_A2

### Relational analysis result of NS_B1_A1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1258950, upper bound: 2861.1257593
time: 0.81 seconds

## BFS NS instance: NS_B1_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -17449.0214844, 21000.9023438, -16638.1113281, 20045.7187500, -37494.7421875, 37639.0078125
1: -1989.5050049, 1812.2502441, -1901.4204102, 1729.5068359, -3719.0117188, 3713.6701660
2: -1170.7468262, 1993.6561279, -1117.1499023, 1902.8555908, -3073.6025391, 3110.8061523
3: -973.4267578, 2080.5781250, -927.5325317, 1987.5738525, -2961.0002441, 3008.1105957
4: -1405.3482666, 1771.4483643, -1340.2596436, 1690.1424561, -3095.4897461, 3111.7080078

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_A1_A2_B1_B1_A1

### Relational analysis result of NS_B1_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259993, upper bound: 2861.1256196
time: 0.78 seconds

## Relational analysis of NS_B1_A1_A2_B1_B1_A2

### Relational analysis result of NS_B1_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259993, upper bound: 2861.1256195
time: 0.79 seconds

## BFS NS instance: NS_B1_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -17449.0214844, 21000.9023438, -17086.6757812, 20682.7519531, -38131.7734375, 38087.5703125
1: -1989.5050049, 1812.2502441, -1962.3250732, 1779.6385498, -3769.1435547, 3774.5751953
2: -1170.7468262, 1993.6561279, -1149.7426758, 1959.7862549, -3130.5327148, 3143.3989258
3: -973.4267578, 2080.5781250, -953.1752319, 2048.5190430, -3021.9455566, 3033.7534180
4: -1405.3482666, 1771.4483643, -1379.7482910, 1739.4660645, -3144.8139648, 3151.1967773

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1_A1_A2_B1_B2_B1

### Relational analysis result of NS_B1_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1255895, upper bound: 2861.1255654
time: 1.01 seconds

## Relational analysis of NS_B1_A1_A2_B1_B2_B2

### Relational analysis result of NS_B1_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1255895, upper bound: 2861.1255654
time: 0.94 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -17198.4296875, 20641.5976562, -17483.0937500, 21051.1035156, -38249.5312500, 38124.6914062
1: -1955.2071533, 1784.1507568, -1994.3098145, 1816.1942139, -3771.4013672, 3778.4599609
2: -1152.4566650, 1962.0009766, -1173.3135986, 1998.1241455, -3150.5808105, 3135.3144531
3: -958.9920654, 2046.4283447, -975.3895874, 2085.3896484, -3044.3815918, 3021.8173828
4: -1383.2012939, 1743.8574219, -1408.4492188, 1775.3115234, -3158.5124512, 3152.3066406

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B1_A1_A2_B2_A1_B1

### Relational analysis result of NS_B1_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259938, upper bound: 2861.1256195
time: 0.77 seconds

## Relational analysis of NS_B1_A1_A2_B2_A1_B2

### Relational analysis result of NS_B1_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259938, upper bound: 2861.1256195
time: 0.78 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -17602.0312500, 21209.1093750, -17483.0937500, 21051.1035156, -38653.1328125, 38692.2031250
1: -2009.4707031, 1828.9714355, -1994.3098145, 1816.1942139, -3825.6647949, 3823.2810059
2: -1181.5650635, 2012.6750488, -1173.3135986, 1998.1241455, -3179.6892090, 3185.9887695
3: -982.0476685, 2100.7414551, -975.3895874, 2085.3896484, -3067.4370117, 3076.1311035
4: -1418.5305176, 1787.8250732, -1408.4492188, 1775.3115234, -3193.8417969, 3196.2744141

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B1_A1_A2_B2_A2_B1

### Relational analysis result of NS_B1_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259938, upper bound: 2861.1256195
time: 0.97 seconds

## Relational analysis of NS_B1_A1_A2_B2_A2_B2

### Relational analysis result of NS_B1_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259938, upper bound: 2861.1256195
time: 0.76 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -17493.9980469, 21219.9140625, -16638.1113281, 20045.7187500, -37539.7031250, 37858.0234375
1: -2013.6370850, 1823.4180908, -1901.4204102, 1729.5068359, -3743.1440430, 3724.8376465
2: -1178.0850830, 2009.3558350, -1117.1499023, 1902.8555908, -3080.9406738, 3126.5058594
3: -976.0397949, 2100.7680664, -927.5325317, 1987.5738525, -2963.6135254, 3028.3005371
4: -1414.1444092, 1782.3868408, -1340.2596436, 1690.1424561, -3104.2861328, 3122.6462402

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_A2_B1_B1_A1_A1

### Relational analysis result of NS_B1_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1260184, upper bound: 2861.1260267
time: 1.08 seconds

## Relational analysis of NS_B1_A2_B1_B1_A1_A2

### Relational analysis result of NS_B1_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1260184, upper bound: 2861.1285569
time: 1.02 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -18062.4589844, 21821.8945312, -16638.1113281, 20045.7187500, -38108.1757812, 38460.0039062
1: -2068.0319824, 1878.7629395, -1901.4204102, 1729.5068359, -3797.5388184, 3780.1826172
2: -1213.8081055, 2069.0183105, -1117.1499023, 1902.8555908, -3116.6635742, 3186.1682129
3: -1007.9404297, 2160.2841797, -927.5325317, 1987.5738525, -2995.5141602, 3087.8166504
4: -1457.6733398, 1836.5395508, -1340.2596436, 1690.1424561, -3147.8156738, 3176.7993164

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_A2_B1_B1_A2_A1

### Relational analysis result of NS_B1_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1260184, upper bound: 2861.1264688
time: 0.83 seconds

## Relational analysis of NS_B1_A2_B1_B1_A2_A2

### Relational analysis result of NS_B1_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1260184, upper bound: 2861.1288376
time: 1.24 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -17493.9980469, 21219.9140625, -17198.4296875, 20641.5976562, -38135.5781250, 38418.3398438
1: -2013.6370850, 1823.4180908, -1955.2071533, 1784.1507568, -3797.7878418, 3778.6250000
2: -1178.0850830, 2009.3558350, -1152.4566650, 1962.0009766, -3140.0859375, 3161.8125000
3: -976.0397949, 2100.7680664, -958.9920654, 2046.4283447, -3022.4675293, 3059.7602539
4: -1414.1444092, 1782.3868408, -1383.2012939, 1743.8574219, -3158.0014648, 3165.5876465

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_A2_B1_B2_A1_A1

### Relational analysis result of NS_B1_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1260184, upper bound: 2861.1260267
time: 0.79 seconds

## Relational analysis of NS_B1_A2_B1_B2_A1_A2

### Relational analysis result of NS_B1_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1260184, upper bound: 2861.1285553
time: 0.92 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -18062.4589844, 21821.8945312, -17198.4296875, 20641.5976562, -38704.0507812, 39020.3203125
1: -2068.0319824, 1878.7629395, -1955.2071533, 1784.1507568, -3852.1826172, 3833.9699707
2: -1213.8081055, 2069.0183105, -1152.4566650, 1962.0009766, -3175.8090820, 3221.4750977
3: -1007.9404297, 2160.2841797, -958.9920654, 2046.4283447, -3054.3684082, 3119.2763672
4: -1457.6733398, 1836.5395508, -1383.2012939, 1743.8574219, -3201.5307617, 3219.7407227

Time for backsubstitution: 2.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_A2_B1_B2_A2_A1

### Relational analysis result of NS_B1_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1260184, upper bound: 2861.1266729
time: 1.25 seconds

## Relational analysis of NS_B1_A2_B1_B2_A2_A2

### Relational analysis result of NS_B1_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1260184, upper bound: 2861.1288935
time: 0.92 seconds

## BFS NS instance: NS_B1_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -17493.9980469, 21219.9140625, -17086.6757812, 20682.7519531, -38176.7343750, 38306.5859375
1: -2013.6370850, 1823.4180908, -1962.3250732, 1779.6385498, -3793.2756348, 3785.7429199
2: -1178.0850830, 2009.3558350, -1149.7426758, 1959.7862549, -3137.8708496, 3159.0986328
3: -976.0397949, 2100.7680664, -953.1752319, 2048.5190430, -3024.5588379, 3053.9433594
4: -1414.1444092, 1782.3868408, -1379.7482910, 1739.4660645, -3153.6103516, 3162.1345215

Time for backsubstitution: 2.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_A2_B2_B1_A1_A1

### Relational analysis result of NS_B1_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1260267, upper bound: 2861.1260267
time: 1.11 seconds

## Relational analysis of NS_B1_A2_B2_B1_A1_A2

### Relational analysis result of NS_B1_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1260267, upper bound: 2861.1285632
time: 1.23 seconds

## BFS NS instance: NS_B1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -18062.4589844, 21821.8945312, -17086.6757812, 20682.7519531, -38745.2070312, 38908.5664062
1: -2068.0319824, 1878.7629395, -1962.3250732, 1779.6385498, -3847.6704102, 3841.0876465
2: -1213.8081055, 2069.0183105, -1149.7426758, 1959.7862549, -3173.5942383, 3218.7609863
3: -1007.9404297, 2160.2841797, -953.1752319, 2048.5190430, -3056.4594727, 3113.4592285
4: -1457.6733398, 1836.5395508, -1379.7482910, 1739.4660645, -3197.1394043, 3216.2873535

Time for backsubstitution: 2.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_A2_B2_B1_A2_A1

### Relational analysis result of NS_B1_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1260267, upper bound: 2861.1264685
time: 0.84 seconds

## Relational analysis of NS_B1_A2_B2_B1_A2_A2

### Relational analysis result of NS_B1_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1260267, upper bound: 2861.1288442
time: 0.79 seconds

## BFS NS instance: NS_B1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -17493.9980469, 21219.9140625, -17652.0742188, 21282.1328125, -38776.1171875, 38871.9882812
1: -2013.6370850, 1823.4180908, -2016.4621582, 1834.7119141, -3848.3491211, 3839.8798828
2: -1178.0850830, 2009.3558350, -1185.3020020, 2019.2042236, -3197.2890625, 3194.6577148
3: -976.0397949, 2100.7680664, -984.9135742, 2107.7641602, -3083.8034668, 3085.6816406
4: -1414.1444092, 1782.3868408, -1423.0430908, 1793.4512939, -3207.5949707, 3205.4291992

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_A2_B2_B2_A1_A1

### Relational analysis result of NS_B1_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1260267, upper bound: 2861.1260267
time: 1.12 seconds

## Relational analysis of NS_B1_A2_B2_B2_A1_A2

### Relational analysis result of NS_B1_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1260267, upper bound: 2861.1285623
time: 0.82 seconds

## BFS NS instance: NS_B1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -18062.4589844, 21821.8945312, -17652.0742188, 21282.1328125, -39344.5859375, 39473.9687500
1: -2068.0319824, 1878.7629395, -2016.4621582, 1834.7119141, -3902.7438965, 3895.2248535
2: -1213.8081055, 2069.0183105, -1185.3020020, 2019.2042236, -3233.0122070, 3254.3203125
3: -1007.9404297, 2160.2841797, -984.9135742, 2107.7641602, -3115.7045898, 3145.1977539
4: -1457.6733398, 1836.5395508, -1423.0430908, 1793.4512939, -3251.1245117, 3259.5820312

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B1_A2_B2_B2_A2_A1

### Relational analysis result of NS_B1_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1260267, upper bound: 2861.1266729
time: 0.86 seconds

## Relational analysis of NS_B1_A2_B2_B2_A2_A2

### Relational analysis result of NS_B1_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1260267, upper bound: 2861.1289012
time: 0.99 seconds

## BFS NS instance: NS_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -16919.5332031, 20453.0253906, -17036.6074219, 20609.5234375, -37529.0429688, 37489.6289062
1: -1940.2930908, 1761.2772217, -1955.3148193, 1773.8862305, -3714.1791992, 3716.5920410
2: -1137.8489990, 1938.8217773, -1145.9982910, 1953.2342529, -3091.0827637, 3084.8200684
3: -943.7532349, 2026.2850342, -950.3074951, 2041.4816895, -2985.2343750, 2976.5925293
4: -1365.2961426, 1721.4396973, -1375.2265625, 1733.8201904, -3099.1162109, 3096.6662598

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1259993
time: 0.81 seconds

## Relational analysis of NS_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1260020
time: 0.77 seconds

## BFS NS instance: NS_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -16919.5332031, 20453.0253906, -17602.0312500, 21209.1093750, -38128.6406250, 38055.0546875
1: -1940.2930908, 1761.2772217, -2009.4707031, 1828.9714355, -3769.2644043, 3770.7480469
2: -1137.8489990, 1938.8217773, -1181.5650635, 2012.6750488, -3150.5234375, 3120.3867188
3: -943.7532349, 2026.2850342, -982.0476685, 2100.7414551, -3044.4946289, 3008.3327637
4: -1365.2961426, 1721.4396973, -1418.5305176, 1787.8250732, -3153.1210938, 3139.9702148

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1259993
time: 0.83 seconds

## Relational analysis of NS_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1260019
time: 0.80 seconds

## BFS NS instance: NS_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -17483.0937500, 21051.1035156, -17036.6074219, 20609.5234375, -38092.6054688, 38087.7070312
1: -1994.3098145, 1816.1942139, -1955.3148193, 1773.8862305, -3768.1960449, 3771.5085449
2: -1173.3135986, 1998.1241455, -1145.9982910, 1953.2342529, -3126.5478516, 3144.1225586
3: -975.3895874, 2085.3896484, -950.3074951, 2041.4816895, -3016.8710938, 3035.6970215
4: -1408.4492188, 1775.3115234, -1375.2265625, 1733.8201904, -3142.2692871, 3150.5378418

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_B1_A2_B1_B1

### Relational analysis result of NS_B2_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1245225, upper bound: 2861.1256541
time: 0.78 seconds

## Relational analysis of NS_B2_A1_B1_A2_B1_B2

### Relational analysis result of NS_B2_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1255653, upper bound: 2861.1259715
time: 1.02 seconds

## BFS NS instance: NS_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -17483.0937500, 21051.1035156, -17602.0312500, 21209.1093750, -38692.2031250, 38653.1328125
1: -1994.3098145, 1816.1942139, -2009.4707031, 1828.9714355, -3823.2810059, 3825.6645508
2: -1173.3135986, 1998.1241455, -1181.5650635, 2012.6750488, -3185.9887695, 3179.6892090
3: -975.3895874, 2085.3896484, -982.0476685, 2100.7414551, -3076.1311035, 3067.4370117
4: -1408.4492188, 1775.3115234, -1418.5305176, 1787.8250732, -3196.2744141, 3193.8417969

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1260754
time: 0.81 seconds

## Relational analysis of NS_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1260788
time: 0.85 seconds

## BFS NS instance: NS_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -17091.5937500, 20688.0625000, -17655.7421875, 21437.7089844, -38529.3046875, 38343.8046875
1: -1962.8297119, 1780.1209717, -2034.5833740, 1840.9606934, -3803.7900391, 3814.7041016
2: -1150.0550537, 1960.3010254, -1189.4440918, 2029.3885498, -3179.4436035, 3149.7446289
3: -953.4449463, 2049.0534668, -985.1143799, 2121.9340820, -3075.3786621, 3034.1677246
4: -1380.1186523, 1739.9373779, -1427.9481201, 1799.6148682, -3179.7333984, 3167.8854980

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1260184
time: 0.84 seconds

## Relational analysis of NS_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1260267
time: 0.91 seconds

## BFS NS instance: NS_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -17091.5937500, 20688.0625000, -18226.6367188, 22041.5761719, -39133.1718750, 38914.6992188
1: -1962.8297119, 1780.1209717, -2089.1479492, 1896.5169678, -3859.3466797, 3869.2687988
2: -1150.0550537, 1960.3010254, -1225.2988281, 2089.2319336, -3239.2871094, 3185.5998535
3: -953.4449463, 2049.0534668, -1017.1414185, 2181.6511230, -3135.0957031, 3066.1943359
4: -1380.1186523, 1739.9373779, -1471.6528320, 1853.8969727, -3234.0151367, 3211.5903320

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1260184
time: 0.75 seconds

## Relational analysis of NS_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1260267
time: 0.88 seconds

## BFS NS instance: NS_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -17656.8007812, 21287.2441406, -17655.7421875, 21437.7089844, -39094.5039062, 38942.9687500
1: -2016.9482422, 1835.1765137, -2034.5833740, 1840.9606934, -3857.9077148, 3869.7595215
2: -1185.6025391, 2019.6976318, -1189.4440918, 2029.3885498, -3214.9909668, 3209.1413574
3: -985.1722412, 2108.2783203, -985.1143799, 2121.9340820, -3107.1064453, 3093.3925781
4: -1423.3986816, 1793.9025879, -1427.9481201, 1799.6148682, -3223.0136719, 3221.8500977

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1264594
time: 0.87 seconds

## Relational analysis of NS_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1264685
time: 0.80 seconds

## BFS NS instance: NS_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -17656.8007812, 21287.2441406, -18226.6367188, 22041.5761719, -39698.3710938, 39513.8710938
1: -2016.9482422, 1835.1765137, -2089.1479492, 1896.5169678, -3913.4643555, 3924.3242188
2: -1185.6025391, 2019.6976318, -1225.2988281, 2089.2319336, -3274.8344727, 3244.9965820
3: -985.1722412, 2108.2783203, -1017.1414185, 2181.6511230, -3166.8232422, 3125.4191895
4: -1423.3986816, 1793.9025879, -1471.6528320, 1853.8969727, -3277.2956543, 3265.5551758

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1260754
time: 1.02 seconds

## Relational analysis of NS_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1266729
time: 0.89 seconds

## BFS NS instance: NS_B2_A2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -17036.6074219, 20609.5234375, -17479.2460938, 21198.1308594, -38234.7226562, 38088.7617188
1: -1955.3148193, 1773.8862305, -2011.5516357, 1821.7159424, -3777.0305176, 3785.4377441
2: -1145.9982910, 1953.2342529, -1176.9786377, 2007.4083252, -3153.4064941, 3130.2128906
3: -950.3074951, 2041.4816895, -975.1929932, 2098.6755371, -3048.9829102, 3016.6745605
4: -1375.2265625, 1733.8201904, -1412.8087158, 1780.7086182, -3155.9348145, 3146.6281738

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_A1_A1_B1_B1

### Relational analysis result of NS_B2_A2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1258524
time: 0.84 seconds

## Relational analysis of NS_B2_A2_A1_A1_B1_B2

### Relational analysis result of NS_B2_A2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1258524
time: 0.97 seconds

## BFS NS instance: NS_B2_A2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -17036.6074219, 20609.5234375, -18047.8847656, 21800.3359375, -38836.9335938, 38657.3984375
1: -1955.3148193, 1773.8862305, -2065.9667969, 1877.0797119, -3832.3945312, 3839.8530273
2: -1145.9982910, 1953.2342529, -1212.7142334, 2067.0957031, -3213.0939941, 3165.9484863
3: -950.3074951, 2041.4816895, -1007.1043091, 2158.2141113, -3108.5214844, 3048.5854492
4: -1375.2265625, 1733.8201904, -1456.3530273, 1834.8884277, -3210.1147461, 3190.1728516

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_A1_A1_B2_B1

### Relational analysis result of NS_B2_A2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1258524
time: 0.91 seconds

## Relational analysis of NS_B2_A2_A1_A1_B2_B2

### Relational analysis result of NS_B2_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1258524
time: 0.81 seconds

## BFS NS instance: NS_B2_A2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -17602.0312500, 21209.1093750, -17479.2460938, 21198.1308594, -38800.1601562, 38688.3554688
1: -2009.4707031, 1828.9714355, -2011.5516357, 1821.7159424, -3831.1862793, 3840.5229492
2: -1181.5650635, 2012.6750488, -1176.9786377, 2007.4083252, -3188.9733887, 3189.6535645
3: -982.0476685, 2100.7414551, -975.1929932, 2098.6755371, -3080.7231445, 3075.9345703
4: -1418.5305176, 1787.8250732, -1412.8087158, 1780.7086182, -3199.2390137, 3200.6335449

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_A1_A2_B1_B1

### Relational analysis result of NS_B2_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1256195
time: 0.83 seconds

## Relational analysis of NS_B2_A2_A1_A2_B1_B2

### Relational analysis result of NS_B2_A2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1256195
time: 0.85 seconds

## BFS NS instance: NS_B2_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -17602.0312500, 21209.1093750, -18047.8847656, 21800.3359375, -39402.3671875, 39256.9921875
1: -2009.4707031, 1828.9714355, -2065.9667969, 1877.0797119, -3886.5502930, 3894.9382324
2: -1181.5650635, 2012.6750488, -1212.7142334, 2067.0957031, -3248.6606445, 3225.3891602
3: -982.0476685, 2100.7414551, -1007.1043091, 2158.2141113, -3140.2614746, 3107.8454590
4: -1418.5305176, 1787.8250732, -1456.3530273, 1834.8884277, -3253.4189453, 3244.1779785

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_A1_A2_B2_B1

### Relational analysis result of NS_B2_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1256195
time: 0.80 seconds

## Relational analysis of NS_B2_A2_A1_A2_B2_B2

### Relational analysis result of NS_B2_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1256196
time: 0.81 seconds

## BFS NS instance: NS_B2_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -17772.5976562, 21573.4316406, -17036.6074219, 20609.5234375, -38382.1015625, 38610.0351562
1: -2047.3842773, 1852.8277588, -1955.3148193, 1773.8862305, -3821.2705078, 3808.1425781
2: -1197.0322266, 2042.5119629, -1145.9982910, 1953.2342529, -3150.2666016, 3188.5095215
3: -991.5979614, 2135.3588867, -950.3074951, 2041.4816895, -3033.0795898, 3085.6662598
4: -1437.1496582, 1811.2342529, -1375.2265625, 1733.8201904, -3170.9694824, 3186.4609375

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_A2_B1_B1_A1

### Relational analysis result of NS_B2_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1284908
time: 0.80 seconds

## Relational analysis of NS_B2_A2_A2_B1_B1_A2

### Relational analysis result of NS_B2_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1286453
time: 0.87 seconds

## BFS NS instance: NS_B2_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -17931.8457031, 21761.9804688, -17602.0312500, 21209.1093750, -39140.9531250, 39364.0117188
1: -2065.2404785, 1869.0228271, -2009.4707031, 1828.9714355, -3894.2119141, 3878.4929199
2: -1207.4073486, 2060.7116699, -1181.5650635, 2012.6750488, -3220.0817871, 3242.2763672
3: -1000.4395142, 2153.8625488, -982.0476685, 2100.7414551, -3101.1809082, 3135.9101562
4: -1449.7738037, 1827.2625732, -1418.5305176, 1787.8250732, -3237.5988770, 3245.7929688

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_A2_B1_B2_A1

### Relational analysis result of NS_B2_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1284908
time: 0.77 seconds

## Relational analysis of NS_B2_A2_A2_B1_B2_A2

### Relational analysis result of NS_B2_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1286453
time: 0.89 seconds

## BFS NS instance: NS_B2_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -17772.5976562, 21573.4316406, -17655.7421875, 21437.7089844, -39210.2968750, 39229.1640625
1: -2047.3842773, 1852.8277588, -2034.5833740, 1840.9606934, -3888.3449707, 3887.4111328
2: -1197.0322266, 2042.5119629, -1189.4440918, 2029.3885498, -3226.4206543, 3231.9550781
3: -991.5979614, 2135.3588867, -985.1143799, 2121.9340820, -3113.5319824, 3120.4731445
4: -1437.1496582, 1811.2342529, -1427.9481201, 1799.6148682, -3236.7646484, 3239.1823730

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_A2_B2_B1_A1

### Relational analysis result of NS_B2_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1357355, upper bound: 2861.1357355
time: 1.02 seconds

## Relational analysis of NS_B2_A2_A2_B2_B1_A2

### Relational analysis result of NS_B2_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1357355, upper bound: 2861.1358748
time: 0.85 seconds

## BFS NS instance: NS_B2_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -17931.8457031, 21761.9804688, -18226.6367188, 22041.5761719, -39973.4218750, 39988.6171875
1: -2065.2404785, 1869.0228271, -2089.1479492, 1896.5169678, -3961.7573242, 3958.1701660
2: -1207.4073486, 2060.7116699, -1225.2988281, 2089.2319336, -3296.6391602, 3286.0104980
3: -1000.4395142, 2153.8625488, -1017.1414185, 2181.6511230, -3182.0905762, 3171.0036621
4: -1449.7738037, 1827.2625732, -1471.6528320, 1853.8969727, -3303.6704102, 3298.9155273

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 41

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_A2_B2_B2_A1

### Relational analysis result of NS_B2_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1358747, upper bound: 2861.1357355
time: 0.93 seconds

## Relational analysis of NS_B2_A2_A2_B2_B2_A2

### Relational analysis result of NS_B2_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1358747, upper bound: 2861.1358747
time: 0.85 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.66 seconds
NS_B1_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1260087, upper bound: 2861.1258524
NS_B1_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1260087, upper bound: 2861.1258524
NS_B1_A1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1257322, upper bound: 2861.1258316
NS_B1_A1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1257303, upper bound: 2861.1251497
NS_B1_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1262219, upper bound: 2861.1258524
NS_B1_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1262219, upper bound: 2861.1258524
NS_B1_A1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1259991, upper bound: 2861.1253033
NS_B1_A1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1258950, upper bound: 2861.1257593
NS_B1_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1259993, upper bound: 2861.1256196
NS_B1_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1259993, upper bound: 2861.1256195
NS_B1_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1255895, upper bound: 2861.1255654
NS_B1_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1255895, upper bound: 2861.1255654
NS_B1_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1259938, upper bound: 2861.1256195
NS_B1_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1259938, upper bound: 2861.1256195
NS_B1_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1259938, upper bound: 2861.1256195
NS_B1_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1259938, upper bound: 2861.1256195
NS_B1_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1260184, upper bound: 2861.1260267
NS_B1_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1260184, upper bound: 2861.1285569
NS_B1_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1260184, upper bound: 2861.1264688
NS_B1_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1260184, upper bound: 2861.1288376
NS_B1_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1260184, upper bound: 2861.1260267
NS_B1_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1260184, upper bound: 2861.1285553
NS_B1_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1260184, upper bound: 2861.1266729
NS_B1_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1260184, upper bound: 2861.1288935
NS_B1_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1260267, upper bound: 2861.1260267
NS_B1_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1260267, upper bound: 2861.1285632
NS_B1_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1260267, upper bound: 2861.1264685
NS_B1_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1260267, upper bound: 2861.1288442
NS_B1_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1260267, upper bound: 2861.1260267
NS_B1_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1260267, upper bound: 2861.1285623
NS_B1_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1260267, upper bound: 2861.1266729
NS_B1_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1260267, upper bound: 2861.1289012
NS_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1259993
NS_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1260020
NS_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1259993
NS_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1260019
NS_B2_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1245225, upper bound: 2861.1256541
NS_B2_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1255653, upper bound: 2861.1259715
NS_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1260754
NS_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1260788
NS_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1260184
NS_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1260267
NS_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1260184
NS_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1260267
NS_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1264594
NS_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1264685
NS_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1260754
NS_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1266729
NS_B2_A2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1258524
NS_B2_A2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1258524
NS_B2_A2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1258524
NS_B2_A2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1258524
NS_B2_A2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1256195
NS_B2_A2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1256195
NS_B2_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1256195
NS_B2_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1256196
NS_B2_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1284908
NS_B2_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1286453
NS_B2_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1284908
NS_B2_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1256196, upper bound: 2861.1286453
NS_B2_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1357355, upper bound: 2861.1357355
NS_B2_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1357355, upper bound: 2861.1358748
NS_B2_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1358747, upper bound: 2861.1357355
NS_B2_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.66
Output dim: 3, lower bound: -2861.1358747, upper bound: 2861.1358747

## BFS NS instance: NS_B1_A1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -16638.1113281, 20045.7187500, -16638.1113281, 20045.7187500, -36683.8281250, 36683.8281250
1: -1901.4204102, 1729.5068359, -1901.4204102, 1729.5068359, -3630.9270020, 3630.9267578
2: -1117.1499023, 1902.8555908, -1117.1499023, 1902.8555908, -3020.0053711, 3020.0053711
3: -927.5325317, 1987.5738525, -927.5325317, 1987.5738525, -2915.1064453, 2915.1064453
4: -1340.2596436, 1690.1424561, -1340.2596436, 1690.1424561, -3030.4018555, 3030.4018555

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A1_B1_A1_B1_A1

### Relational analysis result of NS_B1_A1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259067, upper bound: 2861.1260184
time: 0.98 seconds

## Relational analysis of NS_B1_A1_A1_B1_A1_B1_A2

### Relational analysis result of NS_B1_A1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259067, upper bound: 2861.1259067
time: 0.96 seconds

## BFS NS instance: NS_B1_A1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -16638.1113281, 20045.7187500, -17086.6757812, 20682.7519531, -37320.8593750, 37132.3945312
1: -1901.4204102, 1729.5068359, -1962.3250732, 1779.6385498, -3681.0590820, 3691.8320312
2: -1117.1499023, 1902.8555908, -1149.7426758, 1959.7862549, -3076.9355469, 3052.5981445
3: -927.5325317, 1987.5738525, -953.1752319, 2048.5190430, -2976.0515137, 2940.7490234
4: -1340.2596436, 1690.1424561, -1379.7482910, 1739.4660645, -3079.7255859, 3069.8901367

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A1_B1_A1_B2_B1

### Relational analysis result of NS_B1_A1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1260184, upper bound: 2861.1259067
time: 0.82 seconds

## Relational analysis of NS_B1_A1_A1_B1_A1_B2_B2

### Relational analysis result of NS_B1_A1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259067, upper bound: 2861.1259067
time: 0.81 seconds

## BFS NS instance: NS_B1_A1_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -16504.7519531, 19953.0019531, -16598.3984375, 20084.3750000, -36589.1250000, 36551.3984375
1: -1892.1571045, 1718.0548096, -1905.8564453, 1728.4251709, -3620.5822754, 3623.9111328
2: -1109.6097412, 1892.1370850, -1116.7059326, 1903.5205078, -3013.1298828, 3008.8430176
3: -920.2753296, 1975.9667969, -925.6760254, 1989.5753174, -2909.8500977, 2901.6428223
4: -1332.0823975, 1679.2122803, -1339.8729248, 1689.7469482, -3021.8293457, 3019.0842285

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A1_B1_A2_A1_B1

### Relational analysis result of NS_B1_A1_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1257303, upper bound: 2861.1251495
time: 1.00 seconds

## Relational analysis of NS_B1_A1_A1_B1_A2_A1_B2

### Relational analysis result of NS_B1_A1_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1257303, upper bound: 2861.1251495
time: 0.81 seconds

## BFS NS instance: NS_B1_A1_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -16744.9492188, 20271.0136719, -16712.0937500, 20210.7714844, -36955.7187500, 36983.1093750
1: -1923.4985352, 1743.9824219, -1917.5086670, 1739.9575195, -3663.4560547, 3661.4912109
2: -1126.7829590, 1920.7210693, -1124.1475830, 1915.5810547, -3042.3640137, 3044.8679199
3: -933.9221191, 2007.8454590, -932.0954590, 2002.2322998, -2936.1542969, 2939.9409180
4: -1352.2322998, 1704.8376465, -1348.8913574, 1700.7485352, -3052.9809570, 3053.7290039

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A1_B1_A2_A2_B1

### Relational analysis result of NS_B1_A1_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1257303, upper bound: 2861.1251497
time: 0.80 seconds

## Relational analysis of NS_B1_A1_A1_B1_A2_A2_B2

### Relational analysis result of NS_B1_A1_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1257303, upper bound: 2861.1251497
time: 1.30 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -16638.1113281, 20045.7187500, -17198.4296875, 20641.5976562, -37279.7031250, 37244.1484375
1: -1901.4204102, 1729.5068359, -1955.2071533, 1784.1507568, -3685.5708008, 3684.7138672
2: -1117.1499023, 1902.8555908, -1152.4566650, 1962.0009766, -3079.1506348, 3055.3122559
3: -927.5325317, 1987.5738525, -958.9920654, 2046.4283447, -2973.9604492, 2946.5659180
4: -1340.2596436, 1690.1424561, -1383.2012939, 1743.8574219, -3084.1171875, 3073.3435059

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A1

### Relational analysis result of NS_B1_A1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1258909, upper bound: 2861.1259707
time: 0.84 seconds

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A2

### Relational analysis result of NS_B1_A1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1258910, upper bound: 2861.1259015
time: 0.98 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -16638.1113281, 20045.7187500, -17652.0742188, 21282.1328125, -37920.2421875, 37697.7929688
1: -1901.4204102, 1729.5068359, -2016.4621582, 1834.7119141, -3736.1320801, 3745.9689941
2: -1117.1499023, 1902.8555908, -1185.3020020, 2019.2042236, -3136.3537598, 3088.1577148
3: -927.5325317, 1987.5738525, -984.9135742, 2107.7641602, -3035.2963867, 2972.4873047
4: -1340.2596436, 1690.1424561, -1423.0430908, 1793.4512939, -3133.7109375, 3113.1848145

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_B1

### Relational analysis result of NS_B1_A1_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1264513, upper bound: 2861.1259067
time: 0.81 seconds

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_B2

### Relational analysis result of NS_B1_A1_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1258910, upper bound: 2861.1259016
time: 0.82 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -16762.6464844, 20234.4804688, -17434.7558594, 20984.5898438, -37747.2343750, 37669.2343750
1: -1919.1995850, 1743.5245361, -1987.8900146, 1810.8244629, -3730.0239258, 3731.4145508
2: -1126.3975830, 1918.6373291, -1169.8472900, 1992.0029297, -3118.4003906, 3088.4838867
3: -934.9161987, 2004.9025879, -972.6789551, 2078.9008789, -3013.8168945, 2977.5815430
4: -1351.4876709, 1703.5411377, -1404.2651367, 1769.9580078, -3121.4455566, 3107.8061523

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1_A1_A1_B2_A2_A1_B1

### Relational analysis result of NS_B1_A1_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256102, upper bound: 2861.1252928
time: 0.80 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_A1_B2

### Relational analysis result of NS_B1_A1_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256102, upper bound: 2861.1253033
time: 0.78 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -18947.1855469, 22586.7656250, -17299.8593750, 20823.4667969, -39770.6445312, 39886.6250000
1: -2139.2314453, 1957.8679199, -1972.3718262, 1796.7740479, -3936.0053711, 3930.2392578
2: -1264.3602295, 2148.3093262, -1160.9614258, 1976.4855957, -3240.8454590, 3309.2702637
3: -1055.1048584, 2242.5673828, -965.1990356, 2063.1762695, -3118.2812500, 3207.7661133
4: -1517.7304688, 1909.2319336, -1393.3981934, 1756.2604980, -3273.9907227, 3302.6301270

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1_A1_A1_B2_A2_A2_B1

### Relational analysis result of NS_B1_A1_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256180, upper bound: 2861.1257458
time: 0.89 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_A2_B2

### Relational analysis result of NS_B1_A1_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256180, upper bound: 2861.1257593
time: 0.90 seconds

## BFS NS instance: NS_B1_A1_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -17198.4296875, 20641.5976562, -16638.1113281, 20045.7187500, -37244.1484375, 37279.7070312
1: -1955.2071533, 1784.1507568, -1901.4204102, 1729.5068359, -3684.7138672, 3685.5705566
2: -1152.4566650, 1962.0009766, -1117.1499023, 1902.8555908, -3055.3122559, 3079.1506348
3: -958.9920654, 2046.4283447, -927.5325317, 1987.5738525, -2946.5659180, 2973.9604492
4: -1383.2012939, 1743.8574219, -1340.2596436, 1690.1424561, -3073.3435059, 3084.1171875

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_A1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_B1_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1_A1_A2_B1_B1_A1_B1

### Relational analysis result of NS_B1_A1_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1255895, upper bound: 2861.1255654
time: 1.08 seconds

## Relational analysis of NS_B1_A1_A2_B1_B1_A1_B2

### Relational analysis result of NS_B1_A1_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1258938, upper bound: 2861.1255654
time: 0.77 seconds

## BFS NS instance: NS_B1_A1_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -17602.0312500, 21209.1093750, -16638.1113281, 20045.7187500, -37647.7500000, 37847.2187500
1: -2009.4707031, 1828.9714355, -1901.4204102, 1729.5068359, -3738.9775391, 3730.3918457
2: -1181.5650635, 2012.6750488, -1117.1499023, 1902.8555908, -3084.4206543, 3129.8249512
3: -982.0476685, 2100.7414551, -927.5325317, 1987.5738525, -2969.6215820, 3028.2739258
4: -1418.5305176, 1787.8250732, -1340.2596436, 1690.1424561, -3108.6728516, 3128.0847168

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 34

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_A1_A2_B1_B1_A2_B1

### Relational analysis result of NS_B1_A1_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259990, upper bound: 2861.1256144
time: 0.78 seconds

## Relational analysis of NS_B1_A1_A2_B1_B1_A2_B2

### Relational analysis result of NS_B1_A1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259856, upper bound: 2861.1256144
time: 0.85 seconds

## BFS NS instance: NS_B1_A1_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -17400.7011719, 20934.4003906, -16812.6503906, 20307.6093750, -37708.3046875, 37747.0429688
1: -1983.0864258, 1806.8822021, -1926.1986084, 1749.2698975, -3732.3562012, 3733.0808105
2: -1167.2808838, 1987.5366211, -1130.1383057, 1925.1804199, -3092.4611816, 3117.6748047
3: -970.7165527, 2074.0903320, -937.7825317, 2011.9327393, -2982.6486816, 3011.8728027
4: -1401.1649170, 1766.0976562, -1356.0048828, 1709.1789551, -3110.3432617, 3122.1020508

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B1_A1_A2_B1_B2_B1_A1

### Relational analysis result of NS_B1_A1_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1254429, upper bound: 2861.1245225
time: 1.02 seconds

## Relational analysis of NS_B1_A1_A2_B1_B2_B1_A2

### Relational analysis result of NS_B1_A1_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1254429, upper bound: 2861.1255654
time: 0.83 seconds

## BFS NS instance: NS_B1_A1_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -17265.6679688, 20773.1542969, -18995.8300781, 22658.6035156, -39924.2617188, 39768.9843750
1: -1967.5565186, 1792.8231201, -2146.0832520, 1963.4938965, -3931.0502930, 3938.9062500
2: -1158.3863525, 1972.0074463, -1268.0366211, 2154.6943359, -3313.0803223, 3240.0439453
3: -963.2280884, 2058.3562012, -1057.9158936, 2249.4636230, -3212.6914062, 3116.2714844
4: -1390.2890625, 1752.3890381, -1522.1556396, 1914.7639160, -3305.0529785, 3274.5446777

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B1_A1_A2_B1_B2_B2_A1

### Relational analysis result of NS_B1_A1_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256875, upper bound: 2861.1245225
time: 0.82 seconds

## Relational analysis of NS_B1_A1_A2_B1_B2_B2_A2

### Relational analysis result of NS_B1_A1_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1256875, upper bound: 2861.1255654
time: 0.89 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -17198.4296875, 20641.5976562, -17198.4296875, 20641.5976562, -37840.0234375, 37840.0234375
1: -1955.2071533, 1784.1507568, -1955.2071533, 1784.1507568, -3739.3579102, 3739.3579102
2: -1152.4566650, 1962.0009766, -1152.4566650, 1962.0009766, -3114.4575195, 3114.4575195
3: -958.9920654, 2046.4283447, -958.9920654, 2046.4283447, -3005.4201660, 3005.4201660
4: -1383.2012939, 1743.8574219, -1383.2012939, 1743.8574219, -3127.0585938, 3127.0585938

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B1

### Relational analysis result of NS_B1_A1_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259707, upper bound: 2861.1259215
time: 0.85 seconds

## Relational analysis of NS_B1_A1_A2_B2_A1_B1_B2

### Relational analysis result of NS_B1_A1_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259215, upper bound: 2861.1259215
time: 1.17 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -17198.4296875, 20641.5976562, -17652.0742188, 21282.1328125, -38480.5625000, 38293.6718750
1: -1955.2071533, 1784.1507568, -2016.4621582, 1834.7119141, -3789.9189453, 3800.6127930
2: -1152.4566650, 1962.0009766, -1185.3020020, 2019.2042236, -3171.6606445, 3147.3024902
3: -958.9920654, 2046.4283447, -984.9135742, 2107.7641602, -3066.7561035, 3031.3417969
4: -1383.2012939, 1743.8574219, -1423.0430908, 1793.4512939, -3176.6523438, 3166.9001465

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A2_B2_A1_B2_B1

### Relational analysis result of NS_B1_A1_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259707, upper bound: 2861.1259215
time: 0.98 seconds

## Relational analysis of NS_B1_A1_A2_B2_A1_B2_B2

### Relational analysis result of NS_B1_A1_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259215, upper bound: 2861.1259215
time: 0.86 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -17602.0312500, 21209.1093750, -17198.4296875, 20641.5976562, -38243.6289062, 38407.5351562
1: -2009.4707031, 1828.9714355, -1955.2071533, 1784.1507568, -3793.6213379, 3784.1787109
2: -1181.5650635, 2012.6750488, -1152.4566650, 1962.0009766, -3143.5656738, 3165.1318359
3: -982.0476685, 2100.7414551, -958.9920654, 2046.4283447, -3028.4755859, 3059.7333984
4: -1418.5305176, 1787.8250732, -1383.2012939, 1743.8574219, -3162.3879395, 3171.0263672

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 44

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_A1_A2_B2_A2_B1_B1

### Relational analysis result of NS_B1_A1_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259863, upper bound: 2861.1256144
time: 0.94 seconds

## Relational analysis of NS_B1_A1_A2_B2_A2_B1_B2

### Relational analysis result of NS_B1_A1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1255873, upper bound: 2861.1256145
time: 0.72 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -17602.0312500, 21209.1093750, -17652.0742188, 21282.1328125, -38884.1640625, 38861.1835938
1: -2009.4707031, 1828.9714355, -2016.4621582, 1834.7119141, -3844.1826172, 3845.4335938
2: -1181.5650635, 2012.6750488, -1185.3020020, 2019.2042236, -3200.7687988, 3197.9768066
3: -982.0476685, 2100.7414551, -984.9135742, 2107.7641602, -3089.8117676, 3085.6550293
4: -1418.5305176, 1787.8250732, -1423.0430908, 1793.4512939, -3211.9814453, 3210.8676758

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_A1_A2_B2_A2_B2_B1

### Relational analysis result of NS_B1_A1_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259938, upper bound: 2861.1256196
time: 0.88 seconds

## Relational analysis of NS_B1_A1_A2_B2_A2_B2_B2

### Relational analysis result of NS_B1_A1_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1258253, upper bound: 2861.1256195
time: 0.87 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -17086.6757812, 20682.7519531, -16638.1113281, 20045.7187500, -37132.3906250, 37320.8632812
1: -1962.3250732, 1779.6385498, -1901.4204102, 1729.5068359, -3691.8320312, 3681.0585938
2: -1149.7426758, 1959.7862549, -1117.1499023, 1902.8555908, -3052.5981445, 3076.9355469
3: -953.1752319, 2048.5190430, -927.5325317, 1987.5738525, -2940.7490234, 2976.0515137
4: -1379.7482910, 1739.4660645, -1340.2596436, 1690.1424561, -3069.8901367, 3079.7255859

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A2_B1_B1_A1_A1_A1

### Relational analysis result of NS_B1_A2_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259067, upper bound: 2861.1260267
time: 0.89 seconds

## Relational analysis of NS_B1_A2_B1_B1_A1_A1_A2

### Relational analysis result of NS_B1_A2_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259067, upper bound: 2861.1259266
time: 0.81 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -17655.7421875, 21437.7089844, -16638.1113281, 20045.7187500, -37701.4609375, 38075.8203125
1: -2034.5833740, 1840.9606934, -1901.4204102, 1729.5068359, -3764.0903320, 3742.3808594
2: -1189.4440918, 2029.3885498, -1117.1499023, 1902.8555908, -3092.2998047, 3146.5385742
3: -985.1143799, 2121.9340820, -927.5325317, 1987.5738525, -2972.6882324, 3049.4665527
4: -1427.9481201, 1799.6148682, -1340.2596436, 1690.1424561, -3118.0898438, 3139.8745117

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A2_B1_B1_A1_A2_A1

### Relational analysis result of NS_B1_A2_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259067, upper bound: 2861.1285635
time: 0.83 seconds

## Relational analysis of NS_B1_A2_B1_B1_A1_A2_A2

### Relational analysis result of NS_B1_A2_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259067, upper bound: 2861.1283803
time: 1.06 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -17652.0742188, 21282.1328125, -16638.1113281, 20045.7187500, -37697.7929688, 37920.2421875
1: -2016.4621582, 1834.7119141, -1901.4204102, 1729.5068359, -3745.9689941, 3736.1318359
2: -1185.3020020, 2019.2042236, -1117.1499023, 1902.8555908, -3088.1577148, 3136.3537598
3: -984.9135742, 2107.7641602, -927.5325317, 1987.5738525, -2972.4873047, 3035.2963867
4: -1423.0430908, 1793.4512939, -1340.2596436, 1690.1424561, -3113.1848145, 3133.7109375

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 34

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A2_B1_B1_A2_A1_A1

### Relational analysis result of NS_B1_A2_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259067, upper bound: 2861.1264600
time: 1.09 seconds

## Relational analysis of NS_B1_A2_B1_B1_A2_A1_A2

### Relational analysis result of NS_B1_A2_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259016, upper bound: 2861.1259089
time: 0.89 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -18226.6367188, 22041.5761719, -16638.1113281, 20045.7187500, -38272.3554688, 38679.6875000
1: -2089.1479492, 1896.5169678, -1901.4204102, 1729.5068359, -3818.6547852, 3797.9370117
2: -1225.2988281, 2089.2319336, -1117.1499023, 1902.8555908, -3128.1542969, 3206.3818359
3: -1017.1414185, 2181.6511230, -927.5325317, 1987.5738525, -3004.7150879, 3109.1835938
4: -1471.6528320, 1853.8969727, -1340.2596436, 1690.1424561, -3161.7946777, 3194.1564941

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_A2_B1_B1_A2_A2_B1

### Relational analysis result of NS_B1_A2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259972, upper bound: 2861.1255873
time: 1.19 seconds

## Relational analysis of NS_B1_A2_B1_B1_A2_A2_B2

### Relational analysis result of NS_B1_A2_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259856, upper bound: 2861.1281467
time: 0.93 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -17086.6757812, 20682.7519531, -17198.4296875, 20641.5976562, -37728.2656250, 37881.1796875
1: -1962.3250732, 1779.6385498, -1955.2071533, 1784.1507568, -3746.4755859, 3734.8457031
2: -1149.7426758, 1959.7862549, -1152.4566650, 1962.0009766, -3111.7436523, 3112.2424316
3: -953.1752319, 2048.5190430, -958.9920654, 2046.4283447, -2999.6030273, 3007.5112305
4: -1379.7482910, 1739.4660645, -1383.2012939, 1743.8574219, -3123.6054688, 3122.6674805

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A2_B1_B2_A1_A1_A1

### Relational analysis result of NS_B1_A2_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1258781, upper bound: 2861.1259861
time: 0.86 seconds

## Relational analysis of NS_B1_A2_B1_B2_A1_A1_A2

### Relational analysis result of NS_B1_A2_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1258909, upper bound: 2861.1259217
time: 0.80 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -17655.7421875, 21437.7089844, -17198.4296875, 20641.5976562, -38297.3359375, 38636.1367188
1: -2034.5833740, 1840.9606934, -1955.2071533, 1784.1507568, -3818.7341309, 3796.1679688
2: -1189.4440918, 2029.3885498, -1152.4566650, 1962.0009766, -3151.4445801, 3181.8452148
3: -985.1143799, 2121.9340820, -958.9920654, 2046.4283447, -3031.5424805, 3080.9262695
4: -1427.9481201, 1799.6148682, -1383.2012939, 1743.8574219, -3171.8051758, 3182.8161621

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A2_B1_B2_A1_A2_A1

### Relational analysis result of NS_B1_A2_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1258781, upper bound: 2861.1284700
time: 0.83 seconds

## Relational analysis of NS_B1_A2_B1_B2_A1_A2_A2

### Relational analysis result of NS_B1_A2_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1258909, upper bound: 2861.1283318
time: 0.99 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -17652.0742188, 21282.1328125, -17198.4296875, 20641.5976562, -38293.6718750, 38480.5585938
1: -2016.4621582, 1834.7119141, -1955.2071533, 1784.1507568, -3800.6127930, 3789.9189453
2: -1185.3020020, 2019.2042236, -1152.4566650, 1962.0009766, -3147.3024902, 3171.6606445
3: -984.9135742, 2107.7641602, -958.9920654, 2046.4283447, -3031.3417969, 3066.7561035
4: -1423.0430908, 1793.4512939, -1383.2012939, 1743.8574219, -3166.9001465, 3176.6523438

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A2_B1_B2_A2_A1_A1

### Relational analysis result of NS_B1_A2_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259215, upper bound: 2861.1266508
time: 0.97 seconds

## Relational analysis of NS_B1_A2_B1_B2_A2_A1_A2

### Relational analysis result of NS_B1_A2_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259215, upper bound: 2861.1259357
time: 0.87 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -18226.6367188, 22041.5761719, -17198.4296875, 20641.5976562, -38868.2343750, 39240.0039062
1: -2089.1479492, 1896.5169678, -1955.2071533, 1784.1507568, -3873.2985840, 3851.7241211
2: -1225.2988281, 2089.2319336, -1152.4566650, 1962.0009766, -3187.2998047, 3241.6884766
3: -1017.1414185, 2181.6511230, -958.9920654, 2046.4283447, -3063.5690918, 3140.6430664
4: -1471.6528320, 1853.8969727, -1383.2012939, 1743.8574219, -3215.5102539, 3237.0979004

Time for backsubstitution: 2.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A2_B1_B2_A2_A2_A1

### Relational analysis result of NS_B1_A2_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259215, upper bound: 2861.1288894
time: 0.85 seconds

## Relational analysis of NS_B1_A2_B1_B2_A2_A2_A2

### Relational analysis result of NS_B1_A2_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259215, upper bound: 2861.1284000
time: 0.73 seconds

## BFS NS instance: NS_B1_A2_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -17086.6757812, 20682.7519531, -17086.6757812, 20682.7519531, -37769.4218750, 37769.4257812
1: -1962.3250732, 1779.6385498, -1962.3250732, 1779.6385498, -3741.9636230, 3741.9636230
2: -1149.7426758, 1959.7862549, -1149.7426758, 1959.7862549, -3109.5288086, 3109.5288086
3: -953.1752319, 2048.5190430, -953.1752319, 2048.5190430, -3001.6943359, 3001.6943359
4: -1379.7482910, 1739.4660645, -1379.7482910, 1739.4660645, -3119.2143555, 3119.2143555

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A2_B2_B1_A1_A1_A1

### Relational analysis result of NS_B1_A2_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259266, upper bound: 2861.1260267
time: 1.40 seconds

## Relational analysis of NS_B1_A2_B2_B1_A1_A1_A2

### Relational analysis result of NS_B1_A2_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259266, upper bound: 2861.1259266
time: 0.85 seconds

## BFS NS instance: NS_B1_A2_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -17655.7421875, 21437.7089844, -17086.6757812, 20682.7519531, -38338.4882812, 38524.3828125
1: -2034.5833740, 1840.9606934, -1962.3250732, 1779.6385498, -3814.2219238, 3803.2854004
2: -1189.4440918, 2029.3885498, -1149.7426758, 1959.7862549, -3149.2294922, 3179.1313477
3: -985.1143799, 2121.9340820, -953.1752319, 2048.5190430, -3033.6333008, 3075.1091309
4: -1427.9481201, 1799.6148682, -1379.7482910, 1739.4660645, -3167.4140625, 3179.3632812

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A2_B2_B1_A1_A2_A1

### Relational analysis result of NS_B1_A2_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259266, upper bound: 2861.1285716
time: 0.96 seconds

## Relational analysis of NS_B1_A2_B2_B1_A1_A2_A2

### Relational analysis result of NS_B1_A2_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259266, upper bound: 2861.1283883
time: 0.81 seconds

## BFS NS instance: NS_B1_A2_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -17652.0742188, 21282.1328125, -17086.6757812, 20682.7519531, -38334.8281250, 38368.8046875
1: -2016.4621582, 1834.7119141, -1962.3250732, 1779.6385498, -3796.1005859, 3797.0368652
2: -1185.3020020, 2019.2042236, -1149.7426758, 1959.7862549, -3145.0874023, 3168.9467773
3: -984.9135742, 2107.7641602, -953.1752319, 2048.5190430, -3033.4326172, 3060.9389648
4: -1423.0430908, 1793.4512939, -1379.7482910, 1739.4660645, -3162.5090332, 3173.1989746

Time for backsubstitution: 2.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1_A2_B2_B1_A2_A1_B1

### Relational analysis result of NS_B1_A2_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259864, upper bound: 2861.1258975
time: 0.85 seconds

## Relational analysis of NS_B1_A2_B2_B1_A2_A1_B2

### Relational analysis result of NS_B1_A2_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1259218, upper bound: 2861.1259088
time: 0.90 seconds

## BFS NS instance: NS_B1_A2_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -18226.6367188, 22041.5761719, -17086.6757812, 20682.7519531, -38909.3906250, 39128.2500000
1: -2089.1479492, 1896.5169678, -1962.3250732, 1779.6385498, -3868.7863770, 3858.8420410
2: -1225.2988281, 2089.2319336, -1149.7426758, 1959.7862549, -3185.0847168, 3238.9746094
3: -1017.1414185, 2181.6511230, -953.1752319, 2048.5190430, -3065.6604004, 3134.8261719
4: -1471.6528320, 1853.8969727, -1379.7482910, 1739.4660645, -3211.1188965, 3233.6445312

Time for backsubstitution: 2.70 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.79 + 416.66 = 421.45 seconds
