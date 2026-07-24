## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_7.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 6)
Time budget: 420 seconds
Split limit: 100
Threshold: 1129.173098456097


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-121.0799179, 467.3462524, -121.0799179, 467.3462524, -588.4261475, 588.4261475)
1: (-326.9425354, 1077.0827637, -326.9425354, 1077.0827637, -1404.0250244, 1404.0250244)
2: (-470.5318298, 906.9232178, -470.5318298, 906.9232178, -1377.4550781, 1377.4550781)
3: (-276.3199463, 1150.7871094, -276.3199463, 1150.7871094, -1427.1070557, 1427.1070557)
4: (-433.7043457, 794.8690186, -433.7043457, 794.8690186, -1228.5732422, 1228.5732422)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.93 + 2.47 = 3.40 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -1129.1843903, upper bound: 1129.1843903

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1842345, upper bound: 1129.1840945
time: 0.80 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1840692, upper bound: 1129.1840692
time: 0.92 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.81 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.81
Output dim: 3, lower bound: -1129.1842345, upper bound: 1129.1840945
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.81
Output dim: 3, lower bound: -1129.1840692, upper bound: 1129.1840692

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -118.8301392, 458.5911255, -119.6173096, 461.6656799, -580.4957886, 578.2084351
1: -320.8942871, 1056.7937012, -323.0112000, 1063.9239502, -1384.8179932, 1379.8049316
2: -461.8446350, 889.7792969, -464.8884583, 895.7921753, -1357.6367188, 1354.6673584
3: -271.2226562, 1129.1907959, -273.0078125, 1136.7763672, -1407.9990234, 1402.1986084
4: -425.6771851, 779.9323120, -428.4883423, 785.1677856, -1210.8449707, 1208.4204102

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1840692, upper bound: 1129.1840692
time: 1.02 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1840692, upper bound: 1129.1840692
time: 0.98 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -134.0492096, 515.5631104, -119.1564636, 459.6788940, -593.7280884, 634.7196045
1: -361.5086060, 1186.6343994, -321.7447815, 1059.2071533, -1420.7155762, 1508.3791504
2: -525.0183716, 1000.4839478, -463.1834412, 892.1967773, -1417.2150879, 1463.6673584
3: -306.1023254, 1268.6058350, -271.9434204, 1131.7593994, -1437.8616943, 1540.5490723
4: -483.1453857, 876.7779541, -426.8518372, 781.9419556, -1265.0874023, 1303.6297607

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1840692, upper bound: 1129.1840692
time: 0.99 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1840692, upper bound: 1129.1840692
time: 1.09 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.96 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.96
Output dim: 3, lower bound: -1129.1840692, upper bound: 1129.1840692
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.96
Output dim: 3, lower bound: -1129.1840692, upper bound: 1129.1840692
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.96
Output dim: 3, lower bound: -1129.1840692, upper bound: 1129.1840692
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.96
Output dim: 3, lower bound: -1129.1840692, upper bound: 1129.1840692

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -118.8301392, 458.5911255, -118.8301392, 458.5911255, -577.4212646, 577.4212646
1: -320.8942871, 1056.7937012, -320.8942871, 1056.7937012, -1377.6878662, 1377.6878662
2: -461.8446350, 889.7792969, -461.8446350, 889.7792969, -1351.6236572, 1351.6236572
3: -271.2226562, 1129.1907959, -271.2226562, 1129.1907959, -1400.4133301, 1400.4133301
4: -425.6771851, 779.9323120, -425.6771851, 779.9323120, -1205.6094971, 1205.6094971

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1838856, upper bound: 1129.1837001
time: 0.93 seconds

## Relational analysis of NS_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1837975, upper bound: 1129.1836548
time: 0.82 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -118.8301392, 458.5911255, -134.0492096, 515.5631104, -634.3932495, 592.6403198
1: -320.8942871, 1056.7937012, -361.5086060, 1186.6343994, -1507.5286865, 1418.3022461
2: -461.8446350, 889.7792969, -525.0183716, 1000.4839478, -1462.3283691, 1414.7976074
3: -271.2226562, 1129.1907959, -306.1023254, 1268.6058350, -1539.8283691, 1435.2930908
4: -425.6771851, 779.9323120, -483.1453857, 876.7779541, -1302.4550781, 1263.0776367

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1838382, upper bound: 1129.1837327
time: 0.88 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1837975, upper bound: 1129.1836548
time: 1.35 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -134.0492096, 515.5631104, -118.8301392, 458.5911255, -592.6403198, 634.3932495
1: -361.5086060, 1186.6343994, -320.8942871, 1056.7937012, -1418.3022461, 1507.5286865
2: -525.0183716, 1000.4839478, -461.8446350, 889.7792969, -1414.7976074, 1462.3284912
3: -306.1023254, 1268.6058350, -271.2226562, 1129.1907959, -1435.2930908, 1539.8283691
4: -483.1453857, 876.7779541, -425.6771851, 779.9323120, -1263.0776367, 1302.4550781

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1837054, upper bound: 1129.1836681
time: 1.10 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836275, upper bound: 1129.1836275
time: 1.14 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -134.0492096, 515.5631104, -134.0492096, 515.5631104, -649.6123047, 649.6123047
1: -361.5086060, 1186.6343994, -361.5086060, 1186.6343994, -1548.1430664, 1548.1430664
2: -525.0183716, 1000.4839478, -525.0183716, 1000.4839478, -1525.5023193, 1525.5023193
3: -306.1023254, 1268.6058350, -306.1023254, 1268.6058350, -1574.7081299, 1574.7081299
4: -483.1453857, 876.7779541, -483.1453857, 876.7779541, -1359.9233398, 1359.9233398

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1837054, upper bound: 1129.1836681
time: 0.83 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836275, upper bound: 1129.1836275
time: 0.99 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.36 seconds
NS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 3, lower bound: -1129.1838856, upper bound: 1129.1837001
NS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 3, lower bound: -1129.1837975, upper bound: 1129.1836548
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 3, lower bound: -1129.1838382, upper bound: 1129.1837327
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 3, lower bound: -1129.1837975, upper bound: 1129.1836548
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 3, lower bound: -1129.1837054, upper bound: 1129.1836681
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 3, lower bound: -1129.1836275, upper bound: 1129.1836275
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 3, lower bound: -1129.1837054, upper bound: 1129.1836681
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 3, lower bound: -1129.1836275, upper bound: 1129.1836275

## BFS NS instance: NS_A1_B1_B1

### Backsubstitution after applying NS history:
0: -117.4476547, 453.3078308, -115.9149857, 447.4466248, -564.8942871, 569.2228394
1: -317.1130676, 1044.7041016, -312.9151001, 1031.3139648, -1348.4270020, 1357.6191406
2: -456.3572998, 879.4973145, -450.2749023, 868.0972290, -1324.4541016, 1329.7718506
3: -268.0130615, 1116.2485352, -264.4532776, 1101.9049072, -1369.9179688, 1380.7017822
4: -420.6201172, 770.9227905, -415.0300903, 760.9160767, -1181.5361328, 1185.9528809

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1839659, upper bound: 1129.1839659
time: 1.12 seconds

## Relational analysis of NS_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1839659, upper bound: 1129.1839682
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_B2

### Backsubstitution after applying NS history:
0: -117.7520370, 454.4592285, -133.2097015, 512.6443481, -630.3963623, 587.6688843
1: -317.9105835, 1047.1713867, -358.3791199, 1180.6616211, -1498.5722656, 1405.5505371
2: -457.4949646, 881.8225098, -514.7591553, 999.6190796, -1457.1138916, 1396.5814209
3: -268.6903381, 1118.8548584, -302.8019409, 1259.4532471, -1528.1435547, 1421.6567383
4: -421.6984253, 772.8890381, -475.3560791, 875.1815796, -1296.8800049, 1248.2451172

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1839682, upper bound: 1129.1839659
time: 0.83 seconds

## Relational analysis of NS_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1839682, upper bound: 1129.1839682
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -115.9149857, 447.4466248, -132.7150879, 510.4454041, -626.3604126, 580.1617432
1: -312.9151001, 1031.3139648, -357.8533630, 1174.9393311, -1487.8543701, 1389.1673584
2: -450.2749023, 868.0972290, -519.7418823, 990.5288086, -1440.8032227, 1387.8388672
3: -264.4532776, 1101.9049072, -303.0078125, 1256.0731201, -1520.5263672, 1404.9127197
4: -415.0300903, 760.9160767, -478.2926636, 868.0551147, -1283.0852051, 1239.2084961

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1837975, upper bound: 1129.1836548
time: 1.10 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1837975, upper bound: 1129.1836548
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -133.2097015, 512.6443481, -132.9969025, 511.5062256, -644.7157593, 645.6412354
1: -358.3791199, 1180.6616211, -358.6011047, 1177.1839600, -1535.5631104, 1539.2626953
2: -514.7591553, 999.6190796, -520.8003540, 992.6763306, -1507.4353027, 1520.4193115
3: -302.8019409, 1259.4532471, -303.6358032, 1258.5223389, -1561.3242188, 1563.0889893
4: -475.3560791, 875.1815796, -479.2878723, 869.8842773, -1345.2403564, 1354.4694824

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1837975, upper bound: 1129.1836548
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1837975, upper bound: 1129.1836548
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -132.7150879, 510.4454041, -115.9149857, 447.4466248, -580.1617432, 626.3604126
1: -357.8533630, 1174.9393311, -312.9151001, 1031.3139648, -1389.1673584, 1487.8543701
2: -519.7418823, 990.5288086, -450.2749023, 868.0972290, -1387.8388672, 1440.8032227
3: -303.0078125, 1256.0731201, -264.4532776, 1101.9049072, -1404.9127197, 1520.5263672
4: -478.2926636, 868.0551147, -415.0300903, 760.9160767, -1239.2084961, 1283.0852051

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836548, upper bound: 1129.1837975
time: 1.07 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836548, upper bound: 1129.1837975
time: 0.90 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -132.9969025, 511.5062256, -133.2097015, 512.6443481, -645.6412354, 644.7157593
1: -358.6011047, 1177.1839600, -358.3791199, 1180.6616211, -1539.2626953, 1535.5631104
2: -520.8003540, 992.6763306, -514.7591553, 999.6190796, -1520.4193115, 1507.4353027
3: -303.6358032, 1258.5223389, -302.8019409, 1259.4532471, -1563.0889893, 1561.3242188
4: -479.2878723, 869.8842773, -475.3560791, 875.1815796, -1354.4694824, 1345.2403564

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836548, upper bound: 1129.1837975
time: 0.93 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836548, upper bound: 1129.1837975
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: -132.7150879, 510.4454041, -131.2328796, 504.7539978, -637.4691162, 641.6782837
1: -357.8533630, 1174.9393311, -353.7896729, 1161.9447021, -1519.7980957, 1528.7290039
2: -519.7418823, 990.5288086, -513.8873291, 979.4663696, -1499.2082520, 1504.4160156
3: -303.0078125, 1256.0731201, -299.5716858, 1242.1425781, -1545.1503906, 1555.6447754
4: -478.2926636, 868.0551147, -472.9106750, 858.3517456, -1336.6442871, 1340.9655762

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836275, upper bound: 1129.1836275
time: 0.85 seconds

## Relational analysis of NS_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836275, upper bound: 1129.1836275
time: 1.22 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: -132.9969025, 511.5062256, -148.8550110, 571.2210693, -704.2179565, 660.3610840
1: -358.6011047, 1177.1839600, -400.2461243, 1314.1278076, -1672.7288818, 1577.4300537
2: -520.8003540, 992.6763306, -579.7619019, 1113.6691895, -1634.4694824, 1572.4379883
3: -303.6358032, 1258.5223389, -338.7425537, 1402.5686035, -1706.2041016, 1597.2648926
4: -479.2878723, 869.8842773, -534.3571777, 974.9813843, -1454.2692871, 1404.2410889

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836275, upper bound: 1129.1836275
time: 1.28 seconds

## Relational analysis of NS_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836275, upper bound: 1129.1836275
time: 1.30 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.83 seconds
NS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 3, lower bound: -1129.1839659, upper bound: 1129.1839659
NS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 3, lower bound: -1129.1839659, upper bound: 1129.1839682
NS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 3, lower bound: -1129.1839682, upper bound: 1129.1839659
NS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 3, lower bound: -1129.1839682, upper bound: 1129.1839682
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 3, lower bound: -1129.1837975, upper bound: 1129.1836548
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 3, lower bound: -1129.1837975, upper bound: 1129.1836548
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 3, lower bound: -1129.1837975, upper bound: 1129.1836548
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 3, lower bound: -1129.1837975, upper bound: 1129.1836548
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 3, lower bound: -1129.1836548, upper bound: 1129.1837975
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 3, lower bound: -1129.1836548, upper bound: 1129.1837975
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 3, lower bound: -1129.1836548, upper bound: 1129.1837975
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 3, lower bound: -1129.1836548, upper bound: 1129.1837975
NS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 3, lower bound: -1129.1836275, upper bound: 1129.1836275
NS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 3, lower bound: -1129.1836275, upper bound: 1129.1836275
NS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 3, lower bound: -1129.1836275, upper bound: 1129.1836275
NS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 3, lower bound: -1129.1836275, upper bound: 1129.1836275

## BFS NS instance: NS_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -115.9149857, 447.4466248, -115.9149857, 447.4466248, -563.3616333, 563.3616333
1: -312.9151001, 1031.3139648, -312.9151001, 1031.3139648, -1344.2288818, 1344.2288818
2: -450.2749023, 868.0972290, -450.2749023, 868.0972290, -1318.3715820, 1318.3715820
3: -264.4532776, 1101.9049072, -264.4532776, 1101.9049072, -1366.3581543, 1366.3581543
4: -415.0300903, 760.9160767, -415.0300903, 760.9160767, -1175.9460449, 1175.9461670

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_B1_A1_A1

### Relational analysis result of NS_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1838713, upper bound: 1129.1838559
time: 0.95 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2

### Relational analysis result of NS_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1838728, upper bound: 1129.1839182
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -133.2097015, 512.6443481, -115.9149857, 447.4466248, -580.6563110, 628.5593262
1: -358.3791199, 1180.6616211, -312.9151001, 1031.3139648, -1389.6931152, 1493.5766602
2: -514.7591553, 999.6190796, -450.2749023, 868.0972290, -1382.8559570, 1449.8935547
3: -302.8019409, 1259.4532471, -264.4532776, 1101.9049072, -1404.7067871, 1523.9064941
4: -475.3560791, 875.1815796, -415.0300903, 760.9160767, -1236.2720947, 1290.2116699

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_B1_A2_B1

### Relational analysis result of NS_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1840095, upper bound: 1129.1839864
time: 0.91 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2

### Relational analysis result of NS_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1839157, upper bound: 1129.1839676
time: 1.06 seconds

## BFS NS instance: NS_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -115.9149857, 447.4466248, -133.2097015, 512.6443481, -628.5593262, 580.6563110
1: -312.9151001, 1031.3139648, -358.3791199, 1180.6616211, -1493.5766602, 1389.6931152
2: -450.2749023, 868.0972290, -514.7591553, 999.6190796, -1449.8935547, 1382.8560791
3: -264.4532776, 1101.9049072, -302.8019409, 1259.4532471, -1523.9064941, 1404.7067871
4: -415.0300903, 760.9160767, -475.3560791, 875.1815796, -1290.2116699, 1236.2720947

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_B2_A1_A1

### Relational analysis result of NS_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1839032, upper bound: 1129.1839639
time: 1.09 seconds

## Relational analysis of NS_A1_B1_B2_A1_A2

### Relational analysis result of NS_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1839157, upper bound: 1129.1839157
time: 1.05 seconds

## BFS NS instance: NS_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -133.2097015, 512.6443481, -133.2097015, 512.6443481, -645.8540649, 645.8540649
1: -358.3791199, 1180.6616211, -358.3791199, 1180.6616211, -1539.0407715, 1539.0407715
2: -514.7591553, 999.6190796, -514.7591553, 999.6190796, -1514.3780518, 1514.3780518
3: -302.8019409, 1259.4532471, -302.8019409, 1259.4532471, -1562.2551270, 1562.2551270
4: -475.3560791, 875.1815796, -475.3560791, 875.1815796, -1350.5375977, 1350.5375977

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_B2_A2_A1

### Relational analysis result of NS_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1838621, upper bound: 1129.1838178
time: 1.07 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2

### Relational analysis result of NS_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1838633, upper bound: 1129.1838742
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -115.9149857, 447.4466248, -131.2328796, 504.7539978, -620.6689453, 578.6795044
1: -312.9151001, 1031.3139648, -353.7896729, 1161.9447021, -1474.8598633, 1385.1036377
2: -450.2749023, 868.0972290, -513.8873291, 979.4663696, -1429.7409668, 1381.9842529
3: -264.4532776, 1101.9049072, -299.5716858, 1242.1425781, -1506.5958252, 1401.4765625
4: -415.0300903, 760.9160767, -472.9106750, 858.3517456, -1273.3818359, 1233.8265381

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1_B1

### Relational analysis result of NS_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836587, upper bound: 1129.1835718
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2

### Relational analysis result of NS_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1837493, upper bound: 1129.1835999
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -115.9149857, 447.4466248, -148.8550110, 571.2210693, -687.1359863, 596.3016357
1: -312.9151001, 1031.3139648, -400.2461243, 1314.1278076, -1627.0428467, 1431.5600586
2: -450.2749023, 868.0972290, -579.7619019, 1113.6691895, -1563.9437256, 1447.8587646
3: -264.4532776, 1101.9049072, -338.7425537, 1402.5686035, -1667.0218506, 1440.6474609
4: -415.0300903, 760.9160767, -534.3571777, 974.9813843, -1390.0114746, 1295.2728271

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1838382, upper bound: 1129.1837327
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1837496, upper bound: 1129.1837325
time: 0.95 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -133.2097015, 512.6443481, -131.2328796, 504.7539978, -637.9636841, 643.8771973
1: -358.3791199, 1180.6616211, -353.7896729, 1161.9447021, -1520.3238525, 1534.4512939
2: -514.7591553, 999.6190796, -513.8873291, 979.4663696, -1494.2254639, 1513.5062256
3: -302.8019409, 1259.4532471, -299.5716858, 1242.1425781, -1544.9444580, 1559.0249023
4: -475.3560791, 875.1815796, -472.9106750, 858.3517456, -1333.7077637, 1348.0922852

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1837975, upper bound: 1129.1835687
time: 1.14 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2

### Relational analysis result of NS_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836814, upper bound: 1129.1835703
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -133.2097015, 512.6443481, -148.8550110, 571.2210693, -704.4306030, 661.4993896
1: -358.3791199, 1180.6616211, -400.2461243, 1314.1278076, -1672.5069580, 1580.9077148
2: -514.7591553, 999.6190796, -579.7619019, 1113.6691895, -1628.4282227, 1579.3808594
3: -302.8019409, 1259.4532471, -338.7425537, 1402.5686035, -1705.3704834, 1598.1958008
4: -475.3560791, 875.1815796, -534.3571777, 974.9813843, -1450.3374023, 1409.5385742

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B2_B1

### Relational analysis result of NS_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836192, upper bound: 1129.1834912
time: 1.06 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2

### Relational analysis result of NS_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836943, upper bound: 1129.1835234
time: 1.27 seconds

## BFS NS instance: NS_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -131.2328796, 504.7539978, -115.9149857, 447.4466248, -578.6795044, 620.6689453
1: -353.7896729, 1161.9447021, -312.9151001, 1031.3139648, -1385.1036377, 1474.8598633
2: -513.8873291, 979.4663696, -450.2749023, 868.0972290, -1381.9841309, 1429.7408447
3: -299.5716858, 1242.1425781, -264.4532776, 1101.9049072, -1401.4765625, 1506.5958252
4: -472.9106750, 858.3517456, -415.0300903, 760.9160767, -1233.8265381, 1273.3818359

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_B1_A1_A1

### Relational analysis result of NS_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835718, upper bound: 1129.1836587
time: 0.78 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2

### Relational analysis result of NS_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835999, upper bound: 1129.1837493
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -148.8550110, 571.2210693, -115.9149857, 447.4466248, -596.3016357, 687.1359863
1: -400.2461243, 1314.1278076, -312.9151001, 1031.3139648, -1431.5600586, 1627.0428467
2: -579.7619019, 1113.6691895, -450.2749023, 868.0972290, -1447.8586426, 1563.9437256
3: -338.7425537, 1402.5686035, -264.4532776, 1101.9049072, -1440.6474609, 1667.0218506
4: -534.3571777, 974.9813843, -415.0300903, 760.9160767, -1295.2729492, 1390.0114746

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B1_A2_B1

### Relational analysis result of NS_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1837327, upper bound: 1129.1838382
time: 0.89 seconds

## Relational analysis of NS_A2_B1_B1_A2_B2

### Relational analysis result of NS_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1837325, upper bound: 1129.1837496
time: 0.98 seconds

## BFS NS instance: NS_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -131.2328796, 504.7539978, -133.2097015, 512.6443481, -643.8771973, 637.9636841
1: -353.7896729, 1161.9447021, -358.3791199, 1180.6616211, -1534.4512939, 1520.3238525
2: -513.8873291, 979.4663696, -514.7591553, 999.6190796, -1513.5062256, 1494.2254639
3: -299.5716858, 1242.1425781, -302.8019409, 1259.4532471, -1559.0249023, 1544.9444580
4: -472.9106750, 858.3517456, -475.3560791, 875.1815796, -1348.0922852, 1333.7077637

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_B2_A1_A1

### Relational analysis result of NS_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835523, upper bound: 1129.1837975
time: 0.86 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2

### Relational analysis result of NS_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835556, upper bound: 1129.1836814
time: 1.41 seconds

## BFS NS instance: NS_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -148.8550110, 571.2210693, -133.2097015, 512.6443481, -661.4993896, 704.4306030
1: -400.2461243, 1314.1278076, -358.3791199, 1180.6616211, -1580.9077148, 1672.5069580
2: -579.7619019, 1113.6691895, -514.7591553, 999.6190796, -1579.3808594, 1628.4282227
3: -338.7425537, 1402.5686035, -302.8019409, 1259.4532471, -1598.1958008, 1705.3704834
4: -534.3571777, 974.9813843, -475.3560791, 875.1815796, -1409.5386963, 1450.3374023

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_B2_A2_A1

### Relational analysis result of NS_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834903, upper bound: 1129.1836255
time: 0.95 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2

### Relational analysis result of NS_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835229, upper bound: 1129.1837034
time: 0.93 seconds

## BFS NS instance: NS_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -131.2328796, 504.7539978, -131.2328796, 504.7539978, -635.9868774, 635.9868774
1: -353.7896729, 1161.9447021, -353.7896729, 1161.9447021, -1515.7343750, 1515.7343750
2: -513.8873291, 979.4663696, -513.8873291, 979.4663696, -1493.3536377, 1493.3536377
3: -299.5716858, 1242.1425781, -299.5716858, 1242.1425781, -1541.7142334, 1541.7141113
4: -472.9106750, 858.3517456, -472.9106750, 858.3517456, -1331.2623291, 1331.2623291

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_B1_A1_A1

### Relational analysis result of NS_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835536, upper bound: 1129.1834627
time: 1.10 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2

### Relational analysis result of NS_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835805, upper bound: 1129.1835514
time: 1.11 seconds

## BFS NS instance: NS_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -148.8550110, 571.2210693, -131.2328796, 504.7539978, -653.6089478, 702.4539795
1: -400.2461243, 1314.1278076, -353.7896729, 1161.9447021, -1562.1907959, 1667.9173584
2: -579.7619019, 1113.6691895, -513.8873291, 979.4663696, -1559.2281494, 1627.5565186
3: -338.7425537, 1402.5686035, -299.5716858, 1242.1425781, -1580.8851318, 1702.1402588
4: -534.3571777, 974.9813843, -472.9106750, 858.3517456, -1392.7087402, 1447.8920898

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_B1_A2_B1

### Relational analysis result of NS_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836825, upper bound: 1129.1835453
time: 0.87 seconds

## Relational analysis of NS_A2_B2_B1_A2_B2

### Relational analysis result of NS_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834789, upper bound: 1129.1835098
time: 1.23 seconds

## BFS NS instance: NS_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -131.2328796, 504.7539978, -148.8550110, 571.2210693, -702.4539795, 653.6090088
1: -353.7896729, 1161.9447021, -400.2461243, 1314.1278076, -1667.9173584, 1562.1907959
2: -513.8873291, 979.4663696, -579.7619019, 1113.6691895, -1627.5565186, 1559.2281494
3: -299.5716858, 1242.1425781, -338.7425537, 1402.5686035, -1702.1402588, 1580.8851318
4: -472.9106750, 858.3517456, -534.3571777, 974.9813843, -1447.8920898, 1392.7086182

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_B2_A1_A1

### Relational analysis result of NS_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834789, upper bound: 1129.1836110
time: 1.05 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2

### Relational analysis result of NS_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834789, upper bound: 1129.1834789
time: 1.45 seconds

## BFS NS instance: NS_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -148.8550110, 571.2210693, -148.8550110, 571.2210693, -720.0759277, 720.0759277
1: -400.2461243, 1314.1278076, -400.2461243, 1314.1278076, -1714.3739014, 1714.3739014
2: -579.7619019, 1113.6691895, -579.7619019, 1113.6691895, -1693.4309082, 1693.4309082
3: -338.7425537, 1402.5686035, -338.7425537, 1402.5686035, -1741.3111572, 1741.3111572
4: -534.3571777, 974.9813843, -534.3571777, 974.9813843, -1509.3386230, 1509.3386230

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_B2_A2_B1

### Relational analysis result of NS_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834266, upper bound: 1129.1834759
time: 0.79 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2

### Relational analysis result of NS_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835049, upper bound: 1129.1835049
time: 1.21 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.94 seconds
NS_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 3, lower bound: -1129.1838713, upper bound: 1129.1838559
NS_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 3, lower bound: -1129.1838728, upper bound: 1129.1839182
NS_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 3, lower bound: -1129.1840095, upper bound: 1129.1839864
NS_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 3, lower bound: -1129.1839157, upper bound: 1129.1839676
NS_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 3, lower bound: -1129.1839032, upper bound: 1129.1839639
NS_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 3, lower bound: -1129.1839157, upper bound: 1129.1839157
NS_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 3, lower bound: -1129.1838621, upper bound: 1129.1838178
NS_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 3, lower bound: -1129.1838633, upper bound: 1129.1838742
NS_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 3, lower bound: -1129.1836587, upper bound: 1129.1835718
NS_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 3, lower bound: -1129.1837493, upper bound: 1129.1835999
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 3, lower bound: -1129.1838382, upper bound: 1129.1837327
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 3, lower bound: -1129.1837496, upper bound: 1129.1837325
NS_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 3, lower bound: -1129.1837975, upper bound: 1129.1835687
NS_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 3, lower bound: -1129.1836814, upper bound: 1129.1835703
NS_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 3, lower bound: -1129.1836192, upper bound: 1129.1834912
NS_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 3, lower bound: -1129.1836943, upper bound: 1129.1835234
NS_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 3, lower bound: -1129.1835718, upper bound: 1129.1836587
NS_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 3, lower bound: -1129.1835999, upper bound: 1129.1837493
NS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 3, lower bound: -1129.1837327, upper bound: 1129.1838382
NS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 3, lower bound: -1129.1837325, upper bound: 1129.1837496
NS_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 3, lower bound: -1129.1835523, upper bound: 1129.1837975
NS_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 3, lower bound: -1129.1835556, upper bound: 1129.1836814
NS_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 3, lower bound: -1129.1834903, upper bound: 1129.1836255
NS_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 3, lower bound: -1129.1835229, upper bound: 1129.1837034
NS_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 3, lower bound: -1129.1835536, upper bound: 1129.1834627
NS_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 3, lower bound: -1129.1835805, upper bound: 1129.1835514
NS_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 3, lower bound: -1129.1836825, upper bound: 1129.1835453
NS_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 3, lower bound: -1129.1834789, upper bound: 1129.1835098
NS_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 3, lower bound: -1129.1834789, upper bound: 1129.1836110
NS_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 3, lower bound: -1129.1834789, upper bound: 1129.1834789
NS_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 3, lower bound: -1129.1834266, upper bound: 1129.1834759
NS_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 3, lower bound: -1129.1835049, upper bound: 1129.1835049

## BFS NS instance: NS_A1_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -113.5731506, 438.5350952, -114.5388489, 442.2084351, -555.7816162, 553.0739746
1: -306.5646667, 1010.7988892, -309.1819763, 1019.2537842, -1325.8184814, 1319.9808350
2: -440.9691467, 850.9085083, -444.8090210, 857.9952393, -1298.9641113, 1295.7171631
3: -259.0462341, 1079.9434814, -261.2754517, 1088.9991455, -1348.0454102, 1341.2189941
4: -406.5189819, 745.8069458, -410.0297546, 752.0358276, -1158.5545654, 1155.8366699

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_B1_A1_A1_B1

### Relational analysis result of NS_A1_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1839366, upper bound: 1129.1839366
time: 0.93 seconds

## Relational analysis of NS_A1_B1_B1_A1_A1_B2

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1839366, upper bound: 1129.1839366
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -119.9737930, 463.5579529, -115.1995010, 444.7260132, -564.6998291, 578.7573242
1: -323.5288391, 1069.0716553, -310.9926758, 1025.0661621, -1348.5949707, 1380.0643311
2: -463.5056458, 900.8684692, -447.2903442, 862.8739624, -1326.3795166, 1348.1588135
3: -273.2701416, 1141.8535156, -262.8104553, 1095.2130127, -1368.4831543, 1404.6635742
4: -427.8527527, 789.4682007, -412.3544922, 756.3352661, -1184.1877441, 1201.8227539

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_B1_A1_A2_B1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1839366, upper bound: 1129.1839459
time: 0.99 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1839366, upper bound: 1129.1839459
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -132.0104218, 508.0394287, -113.4394073, 438.0086975, -570.0191040, 621.4788208
1: -355.0975952, 1170.0711670, -306.1546936, 1009.4385376, -1364.5361328, 1476.2258301
2: -510.0824280, 990.6563721, -440.5700378, 849.6433716, -1359.7257080, 1431.2263184
3: -300.0278625, 1248.1959229, -258.7365112, 1078.7250977, -1378.7526855, 1506.9323730
4: -471.0468140, 867.3209229, -406.1148071, 744.6798706, -1215.7266846, 1273.4355469

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836955, upper bound: 1129.1836205
time: 0.84 seconds

## Relational analysis of NS_A1_B1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835607, upper bound: 1129.1835146
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -132.0758362, 508.3059998, -123.5660858, 477.3379211, -609.4137573, 631.8720703
1: -355.3133240, 1170.6740723, -333.1500549, 1100.1533203, -1455.4664307, 1503.8240967
2: -510.3294678, 991.1655884, -476.8092041, 927.9479980, -1438.2774658, 1467.9744873
3: -300.2091980, 1248.7967529, -281.3605652, 1174.4121094, -1474.6213379, 1530.1573486
4: -471.2789612, 867.7894897, -440.2774353, 812.9984741, -1284.2774658, 1308.0668945

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1839032, upper bound: 1129.1839676
time: 1.30 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1839032, upper bound: 1129.1839676
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -113.4394073, 438.0086975, -132.0104218, 508.0394287, -621.4788208, 570.0191040
1: -306.1546936, 1009.4385376, -355.0975952, 1170.0711670, -1476.2258301, 1364.5361328
2: -440.5700378, 849.6433716, -510.0824280, 990.6563721, -1431.2263184, 1359.7257080
3: -258.7365112, 1078.7250977, -300.0278625, 1248.1959229, -1506.9323730, 1378.7528076
4: -406.1148071, 744.6798706, -471.0468140, 867.3209229, -1273.4355469, 1215.7266846

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_B2_A1_A1_A1

### Relational analysis result of NS_A1_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836205, upper bound: 1129.1836955
time: 1.06 seconds

## Relational analysis of NS_A1_B1_B2_A1_A1_A2

### Relational analysis result of NS_A1_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835146, upper bound: 1129.1835607
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -123.5660858, 477.3379211, -132.0758362, 508.3059998, -631.8720703, 609.4137573
1: -333.1500549, 1100.1533203, -355.3133240, 1170.6740723, -1503.8240967, 1455.4664307
2: -476.8092041, 927.9479980, -510.3294678, 991.1655884, -1467.9744873, 1438.2774658
3: -281.3605652, 1174.4121094, -300.2091980, 1248.7967529, -1530.1573486, 1474.6213379
4: -440.2774353, 812.9984741, -471.2789612, 867.7894897, -1308.0668945, 1284.2774658

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1839676, upper bound: 1129.1839032
time: 0.81 seconds

## Relational analysis of NS_A1_B1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1839676, upper bound: 1129.1839157
time: 1.27 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -131.1984406, 504.9515381, -132.0074463, 508.0467834, -639.2451782, 636.9589844
1: -352.9292297, 1162.9967041, -355.1202698, 1170.1070557, -1523.0361328, 1518.1169434
2: -506.8584900, 984.7601318, -510.0332336, 990.7412109, -1497.5997314, 1494.7933350
3: -298.1753845, 1240.5172119, -300.0330505, 1248.1392822, -1546.3145752, 1540.5502930
4: -468.1056824, 862.1494751, -471.0193787, 867.3944092, -1335.5000000, 1333.1685791

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1838083, upper bound: 1129.1838178
time: 1.01 seconds

## Relational analysis of NS_A1_B1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1838083, upper bound: 1129.1838178
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -135.8200226, 523.2105713, -132.0245667, 508.1150818, -643.9350586, 655.2351074
1: -365.1741638, 1206.0606689, -355.1703491, 1170.3348389, -1535.5086670, 1561.2307129
2: -523.3812256, 1020.3242188, -509.9003906, 990.9265137, -1514.3077393, 1530.2246094
3: -308.4322510, 1286.3287354, -300.0643311, 1248.3502197, -1556.7824707, 1586.3930664
4: -483.7073669, 893.5848389, -470.9795227, 867.5670776, -1351.2744141, 1364.5640869

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1838083, upper bound: 1129.1838725
time: 1.30 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1838083, upper bound: 1129.1838742
time: 1.14 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -114.5388489, 442.2084351, -128.8547058, 495.6660156, -610.2048340, 571.0631104
1: -309.1819763, 1019.2537842, -347.3490295, 1141.0483398, -1450.2303467, 1366.6027832
2: -444.8090210, 857.9952393, -504.4868469, 961.9554443, -1406.7641602, 1362.4819336
3: -261.2754517, 1088.9991455, -294.0944824, 1219.7583008, -1481.0336914, 1383.0936279
4: -410.0297546, 752.0358276, -464.2849121, 842.9674683, -1252.9971924, 1216.3205566

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1838355, upper bound: 1129.1836808
time: 0.96 seconds

## Relational analysis of NS_A1_B2_A1_B1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1838355, upper bound: 1129.1837023
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -115.1995010, 444.7260132, -135.1598663, 520.3241577, -635.5236206, 579.8858643
1: -310.9926758, 1025.0661621, -364.0769653, 1198.4763184, -1509.4688721, 1389.1430664
2: -447.2903442, 862.8739624, -526.6557007, 1011.2542114, -1458.5445557, 1389.5294189
3: -262.8104553, 1095.2130127, -308.1332703, 1280.7181396, -1543.5285645, 1403.3461914
4: -412.3544922, 756.3352661, -485.2278137, 886.0899048, -1298.4443359, 1241.5629883

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1838499, upper bound: 1129.1836808
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1838499, upper bound: 1129.1837023
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -114.9067841, 443.3731384, -147.1918335, 564.6683350, -679.5751343, 590.5649414
1: -310.1725464, 1021.8955078, -395.7500610, 1299.0604248, -1609.2329102, 1417.6453857
2: -446.5218201, 860.2679443, -573.3766479, 1100.9854736, -1547.5070801, 1433.6445312
3: -262.1507263, 1091.8857422, -334.9466553, 1386.5037842, -1648.6545410, 1426.8321533
4: -411.5366821, 754.0563354, -528.4531250, 963.9030762, -1375.4396973, 1282.5092773

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1837308, upper bound: 1129.1837205
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1837122, upper bound: 1129.1835556
time: 1.41 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -123.4432449, 475.3384094, -147.3989716, 565.4906006, -688.9337769, 622.7373047
1: -332.6979370, 1096.0496826, -396.2052307, 1301.4459229, -1634.1437988, 1492.2545166
2: -481.2144165, 922.9382935, -573.8978271, 1102.8762207, -1584.0904541, 1496.8361816
3: -281.4545593, 1170.6962891, -335.3395386, 1388.9233398, -1670.3778076, 1506.0357666
4: -443.0777283, 808.8356934, -528.9611816, 965.5245361, -1408.6022949, 1337.7968750

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836313, upper bound: 1129.1837135
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835876, upper bound: 1129.1835179
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -132.0104218, 508.0394287, -128.7663574, 495.3006287, -627.3110352, 636.8057861
1: -355.0975952, 1170.0711670, -347.0697632, 1140.0523682, -1495.1499023, 1517.1408691
2: -510.0824280, 990.6563721, -504.3222046, 961.0289917, -1471.1110840, 1494.9785156
3: -300.0278625, 1248.1959229, -293.9059143, 1218.9370117, -1518.9644775, 1542.1018066
4: -471.0468140, 867.3209229, -464.0914612, 842.1446533, -1313.1911621, 1331.4122314

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 20

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B1_B1_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836990, upper bound: 1129.1835363
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_B2

### Relational analysis result of NS_A1_B2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836990, upper bound: 1129.1835531
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -132.0758362, 508.3059998, -138.0892792, 531.6274414, -663.7032471, 646.3952637
1: -355.3133240, 1170.6740723, -371.9006653, 1223.8272705, -1579.1406250, 1542.5747070
2: -510.3294678, 991.1655884, -537.4407959, 1033.5794678, -1543.9088135, 1528.6063232
3: -300.2091980, 1248.7967529, -314.7073669, 1307.1173096, -1607.3265381, 1563.5041504
4: -471.2789612, 867.7894897, -495.1140747, 905.4372559, -1376.7161865, 1362.9035645

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836767, upper bound: 1129.1835941
time: 1.07 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836767, upper bound: 1129.1835941
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -132.0074463, 508.0467834, -146.8577118, 563.5684814, -695.5759277, 654.9044800
1: -355.1202698, 1170.1070557, -394.8364258, 1296.5168457, -1651.6370850, 1564.9434814
2: -510.0332336, 990.7412109, -571.9429932, 1098.8616943, -1608.8948975, 1562.6840820
3: -300.0330505, 1248.1392822, -334.1492920, 1383.6822510, -1683.7153320, 1582.2885742
4: -471.0193787, 867.3944092, -527.1762695, 962.0034180, -1433.0228271, 1394.5706787

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836160, upper bound: 1129.1834444
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836160, upper bound: 1129.1834912
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -132.0245667, 508.1150818, -151.3416138, 581.2679443, -713.2924805, 659.4565430
1: -355.1703491, 1170.3348389, -406.7064819, 1338.2929688, -1693.4632568, 1577.0410156
2: -509.9003906, 990.9265137, -587.9353027, 1133.3332520, -1643.2335205, 1578.8618164
3: -300.0643311, 1248.3502197, -344.1180115, 1428.1633301, -1728.2275391, 1592.4682617
4: -470.9795227, 867.5670776, -542.2508545, 992.4601440, -1463.4396973, 1409.8177490

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836752, upper bound: 1129.1834545
time: 1.53 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836752, upper bound: 1129.1835234
time: 1.02 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -128.8547058, 495.6660156, -114.5388489, 442.2084351, -571.0631104, 610.2048340
1: -347.3490295, 1141.0483398, -309.1819763, 1019.2537842, -1366.6027832, 1450.2303467
2: -504.4868469, 961.9554443, -444.8090210, 857.9952393, -1362.4819336, 1406.7641602
3: -294.0944824, 1219.7583008, -261.2754517, 1088.9991455, -1383.0936279, 1481.0336914
4: -464.2849121, 842.9674683, -410.0297546, 752.0358276, -1216.3205566, 1252.9971924

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_B1_A1_A1_B1

### Relational analysis result of NS_A2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836808, upper bound: 1129.1838355
time: 1.09 seconds

## Relational analysis of NS_A2_B1_B1_A1_A1_B2

### Relational analysis result of NS_A2_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836808, upper bound: 1129.1838355
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -135.1598663, 520.3241577, -115.1995010, 444.7260132, -579.8858643, 635.5235596
1: -364.0769653, 1198.4763184, -310.9926758, 1025.0661621, -1389.1430664, 1509.4688721
2: -526.6557007, 1011.2542114, -447.2903442, 862.8739624, -1389.5294189, 1458.5445557
3: -308.1332703, 1280.7181396, -262.8104553, 1095.2130127, -1403.3461914, 1543.5285645
4: -485.2278137, 886.0899048, -412.3544922, 756.3352661, -1241.5629883, 1298.4443359

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_B1_A1_A2_B1

### Relational analysis result of NS_A2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836808, upper bound: 1129.1838499
time: 0.97 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2_B2

### Relational analysis result of NS_A2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836808, upper bound: 1129.1838499
time: 0.95 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -147.1918335, 564.6683350, -114.9067841, 443.3731384, -590.5649414, 679.5751343
1: -395.7500610, 1299.0604248, -310.1725464, 1021.8955078, -1417.6453857, 1609.2329102
2: -573.3766479, 1100.9854736, -446.5218201, 860.2679443, -1433.6445312, 1547.5070801
3: -334.9466553, 1386.5037842, -262.1507263, 1091.8857422, -1426.8321533, 1648.6545410
4: -528.4531250, 963.9030762, -411.5366821, 754.0563354, -1282.5092773, 1375.4396973

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_B1_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1837205, upper bound: 1129.1837308
time: 0.96 seconds

## Relational analysis of NS_A2_B1_B1_A2_B1_B2

### Relational analysis result of NS_A2_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835556, upper bound: 1129.1837122
time: 0.90 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -147.3989716, 565.4906006, -123.4432449, 475.3384094, -622.7373047, 688.9337769
1: -396.2052307, 1301.4459229, -332.6979370, 1096.0496826, -1492.2545166, 1634.1437988
2: -573.8978271, 1102.8762207, -481.2144165, 922.9382935, -1496.8361816, 1584.0904541
3: -335.3395386, 1388.9233398, -281.4545593, 1170.6962891, -1506.0357666, 1670.3779297
4: -528.9611816, 965.5245361, -443.0777283, 808.8356934, -1337.7968750, 1408.6022949

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_B1_A2_B2_B1

### Relational analysis result of NS_A2_B1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1837135, upper bound: 1129.1836313
time: 0.98 seconds

## Relational analysis of NS_A2_B1_B1_A2_B2_B2

### Relational analysis result of NS_A2_B1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835179, upper bound: 1129.1835876
time: 1.09 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -128.7663574, 495.3006287, -132.0104218, 508.0394287, -636.8057861, 627.3110352
1: -347.0697632, 1140.0523682, -355.0975952, 1170.0711670, -1517.1408691, 1495.1499023
2: -504.3222046, 961.0289917, -510.0824280, 990.6563721, -1494.9785156, 1471.1112061
3: -293.9059143, 1218.9370117, -300.0278625, 1248.1959229, -1542.1018066, 1518.9645996
4: -464.0914612, 842.1446533, -471.0468140, 867.3209229, -1331.4121094, 1313.1910400

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_B2_A1_A1_A1

### Relational analysis result of NS_A2_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835363, upper bound: 1129.1836990
time: 1.62 seconds

## Relational analysis of NS_A2_B1_B2_A1_A1_A2

### Relational analysis result of NS_A2_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835531, upper bound: 1129.1837692
time: 1.03 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -138.0892792, 531.6274414, -132.0758362, 508.3059998, -646.3952637, 663.7031860
1: -371.9006653, 1223.8272705, -355.3133240, 1170.6740723, -1542.5747070, 1579.1406250
2: -537.4407959, 1033.5794678, -510.3294678, 991.1655884, -1528.6062012, 1543.9088135
3: -314.7073669, 1307.1173096, -300.2091980, 1248.7967529, -1563.5041504, 1607.3265381
4: -495.1140747, 905.4372559, -471.2789612, 867.7894897, -1362.9035645, 1376.7161865

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_B2_A1_A2_B1

### Relational analysis result of NS_A2_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835941, upper bound: 1129.1836767
time: 0.87 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2_B2

### Relational analysis result of NS_A2_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835941, upper bound: 1129.1836814
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -146.8577118, 563.5684814, -132.0074463, 508.0467834, -654.9044800, 695.5759277
1: -394.8364258, 1296.5168457, -355.1202698, 1170.1070557, -1564.9434814, 1651.6370850
2: -571.9429932, 1098.8616943, -510.0332336, 990.7412109, -1562.6840820, 1608.8948975
3: -334.1492920, 1383.6822510, -300.0330505, 1248.1392822, -1582.2885742, 1683.7150879
4: -527.1762695, 962.0034180, -471.0193787, 867.3944092, -1394.5706787, 1433.0228271

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_B2_A2_A1_B1

### Relational analysis result of NS_A2_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834444, upper bound: 1129.1836208
time: 1.13 seconds

## Relational analysis of NS_A2_B1_B2_A2_A1_B2

### Relational analysis result of NS_A2_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834444, upper bound: 1129.1836255
time: 0.94 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -151.3416138, 581.2679443, -132.0245667, 508.1150818, -659.4566040, 713.2924805
1: -406.7064819, 1338.2929688, -355.1703491, 1170.3348389, -1577.0410156, 1693.4632568
2: -587.9353027, 1133.3332520, -509.9003906, 990.9265137, -1578.8618164, 1643.2335205
3: -344.1180115, 1428.1633301, -300.0643311, 1248.3502197, -1592.4682617, 1728.2275391
4: -542.2508545, 992.4601440, -470.9795227, 867.5670776, -1409.8177490, 1463.4396973

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_B2_A2_A2_B1

### Relational analysis result of NS_A2_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834545, upper bound: 1129.1836770
time: 0.83 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2_B2

### Relational analysis result of NS_A2_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834545, upper bound: 1129.1837034
time: 1.21 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -128.8547058, 495.6660156, -129.8309631, 499.3977051, -628.2524414, 625.4968872
1: -347.3490295, 1141.0483398, -349.9926758, 1149.6319580, -1496.9808350, 1491.0410156
2: -504.4868469, 961.9554443, -508.3447266, 969.1431274, -1473.6300049, 1470.3001709
3: -294.0944824, 1219.7583008, -296.3416138, 1228.9569092, -1523.0513916, 1516.0997314
4: -464.2849121, 842.9674683, -467.8234558, 849.2831421, -1313.5678711, 1310.7908936

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_B1_A1_A1_B1

### Relational analysis result of NS_A2_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836614, upper bound: 1129.1836614
time: 0.87 seconds

## Relational analysis of NS_A2_B2_B1_A1_A1_B2

### Relational analysis result of NS_A2_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836614, upper bound: 1129.1836614
time: 1.42 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -135.1598663, 520.3241577, -130.5213165, 502.0404053, -637.2002563, 650.8454590
1: -364.0769653, 1198.4763184, -351.8770142, 1155.6933594, -1519.7702637, 1550.3531494
2: -526.6557007, 1011.2542114, -510.8920593, 974.2266235, -1500.8818359, 1522.1459961
3: -308.1332703, 1280.7181396, -297.9388733, 1235.4743652, -1543.6074219, 1578.6569824
4: -485.2278137, 886.0899048, -470.2327881, 853.7597046, -1338.9875488, 1356.3227539

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_B1_A1_A2_B1

### Relational analysis result of NS_A2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836614, upper bound: 1129.1836858
time: 0.87 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2_B2

### Relational analysis result of NS_A2_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836614, upper bound: 1129.1836858
time: 1.08 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -147.7101593, 566.8114014, -128.7663574, 495.3006287, -643.0108032, 695.5777588
1: -397.1167297, 1303.9444580, -347.0697632, 1140.0523682, -1537.1689453, 1651.0141602
2: -575.3384399, 1105.0701904, -504.3222046, 961.0289917, -1536.3674316, 1609.3923340
3: -336.1045837, 1391.7506104, -293.9059143, 1218.9370117, -1555.0416260, 1685.6564941
4: -530.2733765, 967.4490967, -464.0914612, 842.1446533, -1372.4178467, 1431.5405273

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_B1_A2_B1_B1

### Relational analysis result of NS_A2_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834844, upper bound: 1129.1834692
time: 1.08 seconds

## Relational analysis of NS_A2_B2_B1_A2_B1_B2

### Relational analysis result of NS_A2_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834844, upper bound: 1129.1834958
time: 1.00 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -147.6974792, 566.7754517, -138.0892792, 531.6274414, -679.3249512, 704.8646851
1: -397.0846863, 1303.9301758, -371.9006653, 1223.8272705, -1620.9118652, 1675.8308105
2: -575.2066040, 1105.0330811, -537.4407959, 1033.5794678, -1608.7856445, 1642.4738770
3: -336.0737915, 1391.6756592, -314.7073669, 1307.1173096, -1643.1911621, 1706.3830566
4: -530.1663818, 967.4282227, -495.1140747, 905.4372559, -1435.6036377, 1462.5422363

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834789, upper bound: 1129.1835098
time: 0.86 seconds

## Relational analysis of NS_A2_B2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834789, upper bound: 1129.1835098
time: 1.04 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -128.7663574, 495.3006287, -147.7101593, 566.8114014, -695.5777588, 643.0108032
1: -347.0697632, 1140.0523682, -397.1167297, 1303.9444580, -1651.0141602, 1537.1689453
2: -504.3222046, 961.0289917, -575.3384399, 1105.0701904, -1609.3923340, 1536.3673096
3: -293.9059143, 1218.9370117, -336.1045837, 1391.7506104, -1685.6564941, 1555.0416260
4: -464.0914612, 842.1446533, -530.2733765, 967.4490967, -1431.5405273, 1372.4178467

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_B2_A1_A1_A1

### Relational analysis result of NS_A2_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834692, upper bound: 1129.1834844
time: 0.99 seconds

## Relational analysis of NS_A2_B2_B2_A1_A1_A2

### Relational analysis result of NS_A2_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834958, upper bound: 1129.1835744
time: 1.03 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -138.0892792, 531.6274414, -147.6974792, 566.7754517, -704.8646851, 679.3249512
1: -371.9006653, 1223.8272705, -397.0846863, 1303.9301758, -1675.8308105, 1620.9118652
2: -537.4407959, 1033.5794678, -575.2066040, 1105.0330811, -1642.4738770, 1608.7855225
3: -314.7073669, 1307.1173096, -336.0737915, 1391.6756592, -1706.3830566, 1643.1911621
4: -495.1140747, 905.4372559, -530.1663818, 967.4282227, -1462.5422363, 1435.6036377

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_B2_A1_A2_B1

### Relational analysis result of NS_A2_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835070, upper bound: 1129.1834789
time: 1.22 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2_B2

### Relational analysis result of NS_A2_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835098, upper bound: 1129.1834789
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -147.6707458, 566.6849365, -146.8577118, 563.5684814, -711.2391968, 713.5426025
1: -397.0379639, 1303.6954346, -394.8364258, 1296.5168457, -1693.5548096, 1698.5318604
2: -575.1182861, 1104.8953857, -571.9429932, 1098.8616943, -1673.9799805, 1676.8381348
3: -336.0171814, 1391.3750000, -334.1492920, 1383.6822510, -1719.6992188, 1725.5242920
4: -530.0934448, 967.2914429, -527.1762695, 962.0034180, -1492.0969238, 1494.4677734

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834226, upper bound: 1129.1834226
time: 0.93 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834226, upper bound: 1129.1834759
time: 0.98 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -147.7125702, 566.8500366, -151.3416138, 581.2679443, -728.9803467, 718.1914673
1: -397.1525269, 1304.1882324, -406.7064819, 1338.2929688, -1735.4455566, 1710.8945312
2: -575.0336304, 1105.2674561, -587.9353027, 1133.3332520, -1708.3666992, 1693.2026367
3: -336.1033936, 1391.8819580, -344.1180115, 1428.1633301, -1764.2663574, 1735.9996338
4: -530.0974121, 967.6307373, -542.2508545, 992.4601440, -1522.5576172, 1509.8811035

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834759, upper bound: 1129.1834266
time: 1.20 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834759, upper bound: 1129.1835049
time: 0.94 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.18 seconds
NS_A1_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1839366, upper bound: 1129.1839366
NS_A1_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1839366, upper bound: 1129.1839366
NS_A1_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1839366, upper bound: 1129.1839459
NS_A1_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1839366, upper bound: 1129.1839459
NS_A1_B1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1836955, upper bound: 1129.1836205
NS_A1_B1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1835607, upper bound: 1129.1835146
NS_A1_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1839032, upper bound: 1129.1839676
NS_A1_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1839032, upper bound: 1129.1839676
NS_A1_B1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1836205, upper bound: 1129.1836955
NS_A1_B1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1835146, upper bound: 1129.1835607
NS_A1_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1839676, upper bound: 1129.1839032
NS_A1_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1839676, upper bound: 1129.1839157
NS_A1_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1838083, upper bound: 1129.1838178
NS_A1_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1838083, upper bound: 1129.1838178
NS_A1_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1838083, upper bound: 1129.1838725
NS_A1_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1838083, upper bound: 1129.1838742
NS_A1_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1838355, upper bound: 1129.1836808
NS_A1_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1838355, upper bound: 1129.1837023
NS_A1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1838499, upper bound: 1129.1836808
NS_A1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1838499, upper bound: 1129.1837023
NS_A1_B2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1837308, upper bound: 1129.1837205
NS_A1_B2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1837122, upper bound: 1129.1835556
NS_A1_B2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1836313, upper bound: 1129.1837135
NS_A1_B2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1835876, upper bound: 1129.1835179
NS_A1_B2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1836990, upper bound: 1129.1835363
NS_A1_B2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1836990, upper bound: 1129.1835531
NS_A1_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1836767, upper bound: 1129.1835941
NS_A1_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1836767, upper bound: 1129.1835941
NS_A1_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1836160, upper bound: 1129.1834444
NS_A1_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1836160, upper bound: 1129.1834912
NS_A1_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1836752, upper bound: 1129.1834545
NS_A1_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1836752, upper bound: 1129.1835234
NS_A2_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1836808, upper bound: 1129.1838355
NS_A2_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1836808, upper bound: 1129.1838355
NS_A2_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1836808, upper bound: 1129.1838499
NS_A2_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1836808, upper bound: 1129.1838499
NS_A2_B1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1837205, upper bound: 1129.1837308
NS_A2_B1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1835556, upper bound: 1129.1837122
NS_A2_B1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1837135, upper bound: 1129.1836313
NS_A2_B1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1835179, upper bound: 1129.1835876
NS_A2_B1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1835363, upper bound: 1129.1836990
NS_A2_B1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1835531, upper bound: 1129.1837692
NS_A2_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1835941, upper bound: 1129.1836767
NS_A2_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1835941, upper bound: 1129.1836814
NS_A2_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1834444, upper bound: 1129.1836208
NS_A2_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1834444, upper bound: 1129.1836255
NS_A2_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1834545, upper bound: 1129.1836770
NS_A2_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1834545, upper bound: 1129.1837034
NS_A2_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1836614, upper bound: 1129.1836614
NS_A2_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1836614, upper bound: 1129.1836614
NS_A2_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1836614, upper bound: 1129.1836858
NS_A2_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1836614, upper bound: 1129.1836858
NS_A2_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1834844, upper bound: 1129.1834692
NS_A2_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1834844, upper bound: 1129.1834958
NS_A2_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1834789, upper bound: 1129.1835098
NS_A2_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1834789, upper bound: 1129.1835098
NS_A2_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1834692, upper bound: 1129.1834844
NS_A2_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1834958, upper bound: 1129.1835744
NS_A2_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1835070, upper bound: 1129.1834789
NS_A2_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1835098, upper bound: 1129.1834789
NS_A2_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1834226, upper bound: 1129.1834226
NS_A2_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1834226, upper bound: 1129.1834759
NS_A2_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1834759, upper bound: 1129.1834266
NS_A2_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 3, lower bound: -1129.1834759, upper bound: 1129.1835049

## BFS NS instance: NS_A1_B1_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -113.5731506, 438.5350952, -113.5731506, 438.5350952, -552.1082764, 552.1082764
1: -306.5646667, 1010.7988892, -306.5646667, 1010.7988892, -1317.3635254, 1317.3635254
2: -440.9691467, 850.9085083, -440.9691467, 850.9085083, -1291.8775635, 1291.8775635
3: -259.0462341, 1079.9434814, -259.0462341, 1079.9434814, -1338.9897461, 1338.9897461
4: -406.5189819, 745.8069458, -406.5189819, 745.8069458, -1152.3258057, 1152.3258057

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_B1_A1_A1_B1_A1

### Relational analysis result of NS_A1_B1_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1838620, upper bound: 1129.1838834
time: 0.88 seconds

## Relational analysis of NS_A1_B1_B1_A1_A1_B1_A2

### Relational analysis result of NS_A1_B1_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1838490, upper bound: 1129.1838513
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -113.5731506, 438.5350952, -119.9737930, 463.5579529, -577.1310425, 558.5089111
1: -306.5646667, 1010.7988892, -323.5288391, 1069.0716553, -1375.6363525, 1334.3276367
2: -440.9691467, 850.9085083, -463.5056458, 900.8684692, -1341.8375244, 1314.4140625
3: -259.0462341, 1079.9434814, -273.2701416, 1141.8535156, -1400.8994141, 1353.2136230
4: -406.5189819, 745.8069458, -427.8527527, 789.4682007, -1195.9871826, 1173.6595459

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_B1

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1838671, upper bound: 1129.1838570
time: 1.58 seconds

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_B2

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1838490, upper bound: 1129.1838513
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -119.9737930, 463.5579529, -113.5731506, 438.5350952, -558.5089111, 577.1311035
1: -323.5288391, 1069.0716553, -306.5646667, 1010.7988892, -1334.3276367, 1375.6363525
2: -463.5056458, 900.8684692, -440.9691467, 850.9085083, -1314.4140625, 1341.8376465
3: -273.2701416, 1141.8535156, -259.0462341, 1079.9434814, -1353.2135010, 1400.8995361
4: -427.8527527, 789.4682007, -406.5189819, 745.8069458, -1173.6594238, 1195.9871826

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_B1_A1_A2_B1_A1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1838454, upper bound: 1129.1838818
time: 0.93 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B1_A2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1838374, upper bound: 1129.1838490
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -119.9737930, 463.5579529, -119.9737930, 463.5579529, -583.5316772, 583.5316772
1: -323.5288391, 1069.0716553, -323.5288391, 1069.0716553, -1392.6004639, 1392.6004639
2: -463.5056458, 900.8684692, -463.5056458, 900.8684692, -1364.3740234, 1364.3741455
3: -273.2701416, 1141.8535156, -273.2701416, 1141.8535156, -1415.1235352, 1415.1235352
4: -427.8527527, 789.4682007, -427.8527527, 789.4682007, -1217.3209229, 1217.3209229

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_A1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1838454, upper bound: 1129.1838818
time: 1.00 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_A2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1838374, upper bound: 1129.1838490
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -130.0984497, 500.5218201, -110.3724365, 425.9588928, -556.0573120, 610.8942261
1: -349.9220276, 1152.6434326, -297.8266296, 981.3660889, -1331.2878418, 1450.4700928
2: -502.7316895, 975.9583130, -428.7757874, 826.0550537, -1328.7863770, 1404.7341309
3: -295.6635742, 1229.5811768, -251.7313538, 1048.8815918, -1344.5450439, 1481.3123779
4: -464.2048035, 854.5030518, -395.1599731, 724.0667725, -1188.2714844, 1249.6629639

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_B1_A2_B1_B1_A1

### Relational analysis result of NS_A1_B1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835607, upper bound: 1129.1835146
time: 0.83 seconds

## Relational analysis of NS_A1_B1_B1_A2_B1_B1_A2

### Relational analysis result of NS_A1_B1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835607, upper bound: 1129.1835146
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -130.7279968, 502.9050293, -125.7156754, 483.9006042, -614.6286011, 628.6206055
1: -351.6385803, 1157.7523193, -338.5631409, 1114.8150635, -1466.4534912, 1496.3151855
2: -505.3120422, 980.5355225, -488.4295044, 940.9030762, -1446.2150879, 1468.9650879
3: -297.1364441, 1235.2011719, -286.5468445, 1190.8876953, -1488.0241699, 1521.7480469
4: -466.5552979, 858.5128784, -450.4055176, 824.3803711, -1290.9355469, 1308.9182129

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_B1_A2_B1_B2_A1

### Relational analysis result of NS_A1_B1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835607, upper bound: 1129.1835146
time: 1.01 seconds

## Relational analysis of NS_A1_B1_B1_A2_B1_B2_A2

### Relational analysis result of NS_A1_B1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835607, upper bound: 1129.1835146
time: 0.97 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -131.2923889, 505.2796936, -123.5660858, 477.3379211, -608.6303101, 628.8457642
1: -353.1367188, 1163.7135010, -333.1500549, 1100.1533203, -1453.2900391, 1496.8635254
2: -507.2873230, 985.2848511, -476.8092041, 927.9479980, -1435.2353516, 1462.0938721
3: -298.3707581, 1241.4412842, -281.3605652, 1174.4121094, -1472.7828369, 1522.8017578
4: -468.4702759, 862.6119385, -440.2774353, 812.9984741, -1281.4687500, 1302.8894043

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835708, upper bound: 1129.1835833
time: 0.83 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834346, upper bound: 1129.1834777
time: 1.05 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -138.9382782, 535.2903442, -123.5660858, 477.3379211, -616.2761841, 658.8564453
1: -373.5462952, 1233.4838867, -333.1500549, 1100.1533203, -1473.6995850, 1566.6339111
2: -534.8917236, 1044.2347412, -476.8092041, 927.9479980, -1462.8397217, 1521.0437012
3: -315.4972534, 1314.9692383, -281.3605652, 1174.4121094, -1489.9094238, 1596.3298340
4: -494.4385376, 914.3750610, -440.2774353, 812.9984741, -1307.4370117, 1354.6524658

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835708, upper bound: 1129.1835833
time: 1.02 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834346, upper bound: 1129.1834777
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_B2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -110.3724365, 425.9588928, -130.0984497, 500.5218201, -610.8942261, 556.0573120
1: -297.8266296, 981.3660889, -349.9220276, 1152.6434326, -1450.4700928, 1331.2877197
2: -428.7757874, 826.0550537, -502.7316895, 975.9583130, -1404.7341309, 1328.7863770
3: -251.7313538, 1048.8815918, -295.6635742, 1229.5811768, -1481.3122559, 1344.5451660
4: -395.1599731, 724.0667725, -464.2048035, 854.5030518, -1249.6629639, 1188.2714844

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_B2_A1_A1_A1_B1

### Relational analysis result of NS_A1_B1_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835146, upper bound: 1129.1835607
time: 1.06 seconds

## Relational analysis of NS_A1_B1_B2_A1_A1_A1_B2

### Relational analysis result of NS_A1_B1_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835146, upper bound: 1129.1835607
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -125.7156754, 483.9006042, -130.7279968, 502.9050293, -628.6206055, 614.6286011
1: -338.5631409, 1114.8150635, -351.6385803, 1157.7523193, -1496.3151855, 1466.4534912
2: -488.4295044, 940.9030762, -505.3120422, 980.5355225, -1468.9650879, 1446.2150879
3: -286.5468445, 1190.8876953, -297.1364441, 1235.2011719, -1521.7480469, 1488.0241699
4: -450.4055176, 824.3803711, -466.5552979, 858.5128784, -1308.9182129, 1290.9356689

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_B2_A1_A1_A2_B1

### Relational analysis result of NS_A1_B1_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835146, upper bound: 1129.1835607
time: 0.83 seconds

## Relational analysis of NS_A1_B1_B2_A1_A1_A2_B2

### Relational analysis result of NS_A1_B1_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835146, upper bound: 1129.1835607
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -123.5660858, 477.3379211, -131.2923889, 505.2796936, -628.8457642, 608.6302490
1: -333.1500549, 1100.1533203, -353.1367188, 1163.7135010, -1496.8635254, 1453.2900391
2: -476.8092041, 927.9479980, -507.2873230, 985.2848511, -1462.0939941, 1435.2353516
3: -281.3605652, 1174.4121094, -298.3707581, 1241.4412842, -1522.8017578, 1472.7828369
4: -440.2774353, 812.9984741, -468.4702759, 862.6119385, -1302.8894043, 1281.4686279

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_B2_A1_A2_B1_A1

### Relational analysis result of NS_A1_B1_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835833, upper bound: 1129.1835708
time: 1.02 seconds

## Relational analysis of NS_A1_B1_B2_A1_A2_B1_A2

### Relational analysis result of NS_A1_B1_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834777, upper bound: 1129.1834346
time: 0.94 seconds

## BFS NS instance: NS_A1_B1_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -123.5660858, 477.3379211, -138.9382782, 535.2903442, -658.8564453, 616.2761841
1: -333.1500549, 1100.1533203, -373.5462952, 1233.4838867, -1566.6339111, 1473.6995850
2: -476.8092041, 927.9479980, -534.8917236, 1044.2347412, -1521.0437012, 1462.8397217
3: -281.3605652, 1174.4121094, -315.4972534, 1314.9692383, -1596.3298340, 1489.9094238
4: -440.2774353, 812.9984741, -494.4385376, 914.3750610, -1354.6524658, 1307.4370117

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_B2_A1_A2_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835833, upper bound: 1129.1835708
time: 1.24 seconds

## Relational analysis of NS_A1_B1_B2_A1_A2_B2_A2

### Relational analysis result of NS_A1_B1_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834777, upper bound: 1129.1834346
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -131.1984406, 504.9515381, -131.1984406, 504.9515381, -636.1499023, 636.1499023
1: -352.9292297, 1162.9967041, -352.9292297, 1162.9967041, -1515.9259033, 1515.9259033
2: -506.8584900, 984.7601318, -506.8584900, 984.7601318, -1491.6186523, 1491.6186523
3: -298.1753845, 1240.5172119, -298.1753845, 1240.5172119, -1538.6926270, 1538.6926270
4: -468.1056824, 862.1494751, -468.1056824, 862.1494751, -1330.2546387, 1330.2546387

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_B2_A2_A1_B1_B1

### Relational analysis result of NS_A1_B1_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1837227, upper bound: 1129.1837035
time: 0.89 seconds

## Relational analysis of NS_A1_B1_B2_A2_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836710, upper bound: 1129.1836852
time: 0.97 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -131.1984406, 504.9515381, -135.8200226, 523.2105713, -654.4089966, 640.7715454
1: -352.9292297, 1162.9967041, -365.1741638, 1206.0606689, -1558.9896240, 1528.1708984
2: -506.8584900, 984.7601318, -523.3812256, 1020.3242188, -1527.1827393, 1508.1412354
3: -298.1753845, 1240.5172119, -308.4322510, 1286.3287354, -1584.5039062, 1548.9494629
4: -468.1056824, 862.1494751, -483.7073669, 893.5848389, -1361.6905518, 1345.8564453

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_B2_A2_A1_B2_B1

### Relational analysis result of NS_A1_B1_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1837227, upper bound: 1129.1837035
time: 1.18 seconds

## Relational analysis of NS_A1_B1_B2_A2_A1_B2_B2

### Relational analysis result of NS_A1_B1_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836710, upper bound: 1129.1836852
time: 1.03 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -135.8200226, 523.2105713, -131.1984406, 504.9515381, -640.7715454, 654.4089355
1: -365.1741638, 1206.0606689, -352.9292297, 1162.9967041, -1528.1708984, 1558.9898682
2: -523.3812256, 1020.3242188, -506.8584900, 984.7601318, -1508.1413574, 1527.1827393
3: -308.4322510, 1286.3287354, -298.1753845, 1240.5172119, -1548.9494629, 1584.5039062
4: -483.7073669, 893.5848389, -468.1056824, 862.1494751, -1345.8563232, 1361.6905518

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_B2_A2_A2_B1_A1

### Relational analysis result of NS_A1_B1_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836803, upper bound: 1129.1837463
time: 0.91 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2_B1_A2

### Relational analysis result of NS_A1_B1_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836710, upper bound: 1129.1837002
time: 1.21 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -135.8200226, 523.2105713, -135.8200226, 523.2105713, -659.0305786, 659.0305786
1: -365.1741638, 1206.0606689, -365.1741638, 1206.0606689, -1571.2346191, 1571.2346191
2: -523.3812256, 1020.3242188, -523.3812256, 1020.3242188, -1543.7054443, 1543.7054443
3: -308.4322510, 1286.3287354, -308.4322510, 1286.3287354, -1594.7609863, 1594.7609863
4: -483.7073669, 893.5848389, -483.7073669, 893.5848389, -1377.2922363, 1377.2922363

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_A1

### Relational analysis result of NS_A1_B1_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836803, upper bound: 1129.1837476
time: 0.85 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836710, upper bound: 1129.1837002
time: 1.04 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -113.5731506, 438.5350952, -128.8547058, 495.6660156, -609.2391357, 567.3897705
1: -306.5646667, 1010.7988892, -347.3490295, 1141.0483398, -1447.6130371, 1358.1479492
2: -440.9691467, 850.9085083, -504.4868469, 961.9554443, -1402.9244385, 1355.3952637
3: -259.0462341, 1079.9434814, -294.0944824, 1219.7583008, -1478.8044434, 1374.0378418
4: -406.5189819, 745.8069458, -464.2849121, 842.9674683, -1249.4863281, 1210.0917969

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_B1_A1_A1

### Relational analysis result of NS_A1_B2_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1838400, upper bound: 1129.1836939
time: 1.23 seconds

## Relational analysis of NS_A1_B2_A1_B1_B1_A1_A2

### Relational analysis result of NS_A1_B2_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1837450, upper bound: 1129.1836939
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -119.9737930, 463.5579529, -128.8547058, 495.6660156, -615.6397705, 592.4125366
1: -323.5288391, 1069.0716553, -347.3490295, 1141.0483398, -1464.5771484, 1416.4205322
2: -463.5056458, 900.8684692, -504.4868469, 961.9554443, -1425.4609375, 1405.3553467
3: -273.2701416, 1141.8535156, -294.0944824, 1219.7583008, -1493.0284424, 1435.9477539
4: -427.8527527, 789.4682007, -464.2849121, 842.9674683, -1270.8200684, 1253.7531738

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_B1_A2_A1

### Relational analysis result of NS_A1_B2_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1838400, upper bound: 1129.1837087
time: 0.96 seconds

## Relational analysis of NS_A1_B2_A1_B1_B1_A2_A2

### Relational analysis result of NS_A1_B2_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1837450, upper bound: 1129.1837087
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -113.5731506, 438.5350952, -135.1598663, 520.3241577, -633.8973389, 573.6949463
1: -306.5646667, 1010.7988892, -364.0769653, 1198.4763184, -1505.0410156, 1374.8758545
2: -440.9691467, 850.9085083, -526.6557007, 1011.2542114, -1452.2233887, 1377.5638428
3: -259.0462341, 1079.9434814, -308.1332703, 1280.7181396, -1539.7644043, 1388.0765381
4: -406.5189819, 745.8069458, -485.2278137, 886.0899048, -1292.6088867, 1231.0346680

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_B2_A1_A1

### Relational analysis result of NS_A1_B2_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1838355, upper bound: 1129.1836808
time: 1.33 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2_A1_A2

### Relational analysis result of NS_A1_B2_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1837450, upper bound: 1129.1836808
time: 1.24 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -119.9737930, 463.5579529, -135.1598663, 520.3241577, -640.2979126, 598.7177734
1: -323.5288391, 1069.0716553, -364.0769653, 1198.4763184, -1522.0050049, 1433.1486816
2: -463.5056458, 900.8684692, -526.6557007, 1011.2542114, -1474.7598877, 1427.5240479
3: -273.2701416, 1141.8535156, -308.1332703, 1280.7181396, -1553.9882812, 1449.9864502
4: -427.8527527, 789.4682007, -485.2278137, 886.0899048, -1313.9426270, 1274.6960449

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1838355, upper bound: 1129.1837023
time: 1.33 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1837450, upper bound: 1129.1837023
time: 1.10 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -112.3522720, 433.6424866, -146.0508270, 560.2783203, -672.6306152, 579.6932983
1: -303.1978149, 999.3215332, -392.6341553, 1288.9143066, -1592.1120605, 1391.9556885
2: -436.4948730, 841.2479248, -568.9746704, 1092.4127197, -1528.9074707, 1410.2225342
3: -256.2513733, 1067.9714355, -332.3207397, 1375.7280273, -1631.9793701, 1400.2922363
4: -402.3291626, 737.3126831, -524.3908691, 956.3925781, -1358.7216797, 1261.7036133

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A1_A1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836799, upper bound: 1129.1836397
time: 0.88 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_A1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836754, upper bound: 1129.1836577
time: 1.08 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -122.5544739, 473.2990417, -146.0388184, 560.2506104, -682.8049927, 619.3378296
1: -330.3968811, 1090.8793945, -392.6033020, 1288.9180908, -1619.3146973, 1483.4825439
2: -472.9741211, 920.2214355, -568.8447876, 1092.3839111, -1565.3580322, 1489.0659180
3: -279.0466614, 1164.4838867, -332.2908936, 1375.6737061, -1654.7203369, 1496.7747803
4: -436.7086487, 806.2061157, -524.2844238, 956.3779907, -1393.0866699, 1330.4903564

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1837122, upper bound: 1129.1835523
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1837122, upper bound: 1129.1835556
time: 1.41 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -120.9983902, 466.0179443, -146.2653198, 561.1360474, -682.1344604, 612.2832642
1: -326.0514832, 1074.3497314, -393.1139526, 1291.3802490, -1617.4317627, 1467.4636230
2: -471.7249146, 904.6795044, -569.5173950, 1094.3728027, -1566.0975342, 1474.1968994
3: -275.8404541, 1147.6873779, -332.7311401, 1378.2263184, -1654.0665283, 1480.4184570
4: -434.3334656, 792.7854004, -524.9186401, 958.0690918, -1392.4025879, 1317.7041016

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835876, upper bound: 1129.1835179
time: 1.15 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835876, upper bound: 1129.1835179
time: 1.14 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -130.3862457, 502.5742493, -146.2314148, 561.0218506, -691.4080811, 648.8056641
1: -351.0736694, 1158.4904785, -393.0219116, 1291.1850586, -1642.2587891, 1551.5124512
2: -505.2575989, 977.6444092, -569.2935791, 1094.1822510, -1599.4398193, 1546.9379883
3: -296.8209839, 1236.3240967, -332.6503296, 1377.9665527, -1674.7874756, 1568.9743652
4: -465.8970642, 856.4816284, -524.7306519, 957.9172363, -1423.8140869, 1381.2122803

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835876, upper bound: 1129.1835179
time: 1.12 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835876, upper bound: 1129.1835179
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -130.8279419, 503.5150757, -126.4143906, 486.3462524, -617.1741943, 629.9293823
1: -351.8941040, 1159.6789551, -340.7070312, 1119.4364014, -1471.3305664, 1500.3859863
2: -505.4384155, 981.9180298, -495.0114746, 943.7409668, -1449.1791992, 1476.9294434
3: -297.3092346, 1237.0557861, -288.4903259, 1196.8377686, -1494.1469727, 1525.5460205
4: -466.7855225, 859.6569214, -455.5537109, 826.9609375, -1293.7464600, 1315.2106934

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 20

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836904, upper bound: 1129.1833883
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836904, upper bound: 1129.1835363
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -130.7947083, 503.3945312, -132.8479767, 511.4957581, -642.2904663, 636.2424316
1: -351.8052979, 1159.4824219, -357.7890625, 1178.0196533, -1529.8249512, 1517.2714844
2: -505.0921021, 981.7504272, -517.7007446, 993.9829102, -1499.0747070, 1499.4511719
3: -297.2156067, 1236.8096924, -302.8157043, 1259.0712891, -1556.2868652, 1539.6253662
4: -466.5547791, 859.5215454, -476.9688110, 870.9146729, -1337.4693604, 1336.4903564

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 20

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B1_B1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1837306, upper bound: 1129.1833889
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1837306, upper bound: 1129.1835531
time: 1.21 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -131.2923889, 505.2796936, -138.0892792, 531.6274414, -662.9196777, 643.3689575
1: -353.1367188, 1163.7135010, -371.9006653, 1223.8272705, -1576.9639893, 1535.6141357
2: -507.2873230, 985.2848511, -537.4407959, 1033.5794678, -1540.8665771, 1522.7255859
3: -298.3707581, 1241.4412842, -314.7073669, 1307.1173096, -1605.4880371, 1556.1486816
4: -468.4702759, 862.6119385, -495.1140747, 905.4372559, -1373.9074707, 1357.7259521

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1833871, upper bound: 1129.1833772
time: 0.99 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836338, upper bound: 1129.1835238
time: 1.03 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -138.9382782, 535.2903442, -138.0892792, 531.6274414, -670.5657349, 673.3795166
1: -373.5462952, 1233.4838867, -371.9006653, 1223.8272705, -1597.3735352, 1605.3845215
2: -534.8917236, 1044.2347412, -537.4407959, 1033.5794678, -1568.4711914, 1581.6755371
3: -315.4972534, 1314.9692383, -314.7073669, 1307.1173096, -1622.6145020, 1629.6766357
4: -494.4385376, 914.3750610, -495.1140747, 905.4372559, -1399.8757324, 1409.4890137

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1833871, upper bound: 1129.1833772
time: 1.18 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836338, upper bound: 1129.1835238
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -131.1984406, 504.9515381, -146.8577118, 563.5684814, -694.7667847, 651.8092651
1: -352.9292297, 1162.9967041, -394.8364258, 1296.5168457, -1649.4460449, 1557.8331299
2: -506.8584900, 984.7601318, -571.9429932, 1098.8616943, -1605.7202148, 1556.7030029
3: -298.1753845, 1240.5172119, -334.1492920, 1383.6822510, -1681.8574219, 1574.6665039
4: -468.1056824, 862.1494751, -527.1762695, 962.0034180, -1430.1091309, 1389.3253174

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_B1_A1_A1

### Relational analysis result of NS_A1_B2_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836160, upper bound: 1129.1834444
time: 1.31 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_A1_A2

### Relational analysis result of NS_A1_B2_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835075, upper bound: 1129.1834444
time: 0.88 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -135.8200226, 523.2105713, -146.8577118, 563.5684814, -699.3884888, 670.0682983
1: -365.1741638, 1206.0606689, -394.8364258, 1296.5168457, -1661.6910400, 1600.8969727
2: -523.3812256, 1020.3242188, -571.9429932, 1098.8616943, -1622.2429199, 1592.2672119
3: -308.4322510, 1286.3287354, -334.1492920, 1383.6822510, -1692.1143799, 1620.4780273
4: -483.7073669, 893.5848389, -527.1762695, 962.0034180, -1445.7108154, 1420.7611084

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A1

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836160, upper bound: 1129.1834912
time: 1.01 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A2

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835075, upper bound: 1129.1834912
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -131.1984406, 504.9515381, -151.3416138, 581.2679443, -712.4662476, 656.2930908
1: -352.9292297, 1162.9967041, -406.7064819, 1338.2929688, -1691.2221680, 1569.7031250
2: -506.8584900, 984.7601318, -587.9353027, 1133.3332520, -1640.1916504, 1572.6954346
3: -298.1753845, 1240.5172119, -344.1180115, 1428.1633301, -1726.3385010, 1584.6352539
4: -468.1056824, 862.1494751, -542.2508545, 992.4601440, -1460.5657959, 1404.3995361

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_B2_A1_A1

### Relational analysis result of NS_A1_B2_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836160, upper bound: 1129.1834545
time: 1.34 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2_A1_A2

### Relational analysis result of NS_A1_B2_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835075, upper bound: 1129.1834544
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -135.8200226, 523.2105713, -151.3416138, 581.2679443, -717.0878906, 674.5521240
1: -365.1741638, 1206.0606689, -406.7064819, 1338.2929688, -1703.4670410, 1612.7669678
2: -523.3812256, 1020.3242188, -587.9353027, 1133.3332520, -1656.7142334, 1608.2595215
3: -308.4322510, 1286.3287354, -344.1180115, 1428.1633301, -1736.5955811, 1630.4466553
4: -483.7073669, 893.5848389, -542.2508545, 992.4601440, -1476.1674805, 1435.8353271

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836160, upper bound: 1129.1835234
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835075, upper bound: 1129.1835234
time: 0.95 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -128.8547058, 495.6660156, -113.5731506, 438.5350952, -567.3897705, 609.2391357
1: -347.3490295, 1141.0483398, -306.5646667, 1010.7988892, -1358.1479492, 1447.6130371
2: -504.4868469, 961.9554443, -440.9691467, 850.9085083, -1355.3953857, 1402.9244385
3: -294.0944824, 1219.7583008, -259.0462341, 1079.9434814, -1374.0378418, 1478.8045654
4: -464.2849121, 842.9674683, -406.5189819, 745.8069458, -1210.0917969, 1249.4863281

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_B1

### Relational analysis result of NS_A2_B1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836939, upper bound: 1129.1838400
time: 0.87 seconds

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_B2

### Relational analysis result of NS_A2_B1_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836939, upper bound: 1129.1837450
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -128.8547058, 495.6660156, -119.9737930, 463.5579529, -592.4125366, 615.6397095
1: -347.3490295, 1141.0483398, -323.5288391, 1069.0716553, -1416.4206543, 1464.5771484
2: -504.4868469, 961.9554443, -463.5056458, 900.8684692, -1405.3553467, 1425.4609375
3: -294.0944824, 1219.7583008, -273.2701416, 1141.8535156, -1435.9477539, 1493.0284424
4: -464.2849121, 842.9674683, -427.8527527, 789.4682007, -1253.7531738, 1270.8200684

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B1_A1_A1_B2_B1

### Relational analysis result of NS_A2_B1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836939, upper bound: 1129.1838400
time: 1.31 seconds

## Relational analysis of NS_A2_B1_B1_A1_A1_B2_B2

### Relational analysis result of NS_A2_B1_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836939, upper bound: 1129.1837450
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -135.1598663, 520.3241577, -113.5731506, 438.5350952, -573.6949463, 633.8973389
1: -364.0769653, 1198.4763184, -306.5646667, 1010.7988892, -1374.8758545, 1505.0410156
2: -526.6557007, 1011.2542114, -440.9691467, 850.9085083, -1377.5638428, 1452.2233887
3: -308.1332703, 1280.7181396, -259.0462341, 1079.9434814, -1388.0765381, 1539.7644043
4: -485.2278137, 886.0899048, -406.5189819, 745.8069458, -1231.0346680, 1292.6088867

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836808, upper bound: 1129.1838499
time: 0.88 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_B2

### Relational analysis result of NS_A2_B1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836808, upper bound: 1129.1837710
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -135.1598663, 520.3241577, -119.9737930, 463.5579529, -598.7177734, 640.2979736
1: -364.0769653, 1198.4763184, -323.5288391, 1069.0716553, -1433.1486816, 1522.0050049
2: -526.6557007, 1011.2542114, -463.5056458, 900.8684692, -1427.5239258, 1474.7598877
3: -308.1332703, 1280.7181396, -273.2701416, 1141.8535156, -1449.9864502, 1553.9882812
4: -485.2278137, 886.0899048, -427.8527527, 789.4682007, -1274.6960449, 1313.9426270

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B1_A1_A2_B2_B1

### Relational analysis result of NS_A2_B1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836808, upper bound: 1129.1838499
time: 0.92 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2_B2_B2

### Relational analysis result of NS_A2_B1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836808, upper bound: 1129.1837710
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -146.0508270, 560.2783203, -112.3522720, 433.6424866, -579.6932983, 672.6306152
1: -392.6341553, 1288.9143066, -303.1978149, 999.3215332, -1391.9556885, 1592.1120605
2: -568.9746704, 1092.4127197, -436.4948730, 841.2479248, -1410.2225342, 1528.9074707
3: -332.3207397, 1375.7280273, -256.2513733, 1067.9714355, -1400.2922363, 1631.9793701
4: -524.3908691, 956.3925781, -402.3291626, 737.3126831, -1261.7036133, 1358.7216797

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_B1_A2_B1_B1_B1

### Relational analysis result of NS_A2_B1_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836397, upper bound: 1129.1836799
time: 1.39 seconds

## Relational analysis of NS_A2_B1_B1_A2_B1_B1_B2

### Relational analysis result of NS_A2_B1_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1836577, upper bound: 1129.1836754
time: 0.90 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -146.0388184, 560.2506104, -122.5544739, 473.2990417, -619.3378296, 682.8049316
1: -392.6033020, 1288.9180908, -330.3968811, 1090.8793945, -1483.4825439, 1619.3148193
2: -568.8447876, 1092.3839111, -472.9741211, 920.2214355, -1489.0660400, 1565.3580322
3: -332.2908936, 1375.6737061, -279.0466614, 1164.4838867, -1496.7747803, 1654.7203369
4: -524.2844238, 956.3779907, -436.7086487, 806.2061157, -1330.4903564, 1393.0866699

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_B1_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835523, upper bound: 1129.1837122
time: 0.95 seconds

## Relational analysis of NS_A2_B1_B1_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835523, upper bound: 1129.1837122
time: 1.09 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -146.2653198, 561.1360474, -120.9983902, 466.0179443, -612.2832642, 682.1344604
1: -393.1139526, 1291.3802490, -326.0514832, 1074.3497314, -1467.4636230, 1617.4317627
2: -569.5173950, 1094.3728027, -471.7249146, 904.6795044, -1474.1968994, 1566.0975342
3: -332.7311401, 1378.2263184, -275.8404541, 1147.6873779, -1480.4184570, 1654.0665283
4: -524.9186401, 958.0690918, -434.3334656, 792.7854004, -1317.7041016, 1392.4025879

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_B1_A2_B2_B1_A1

### Relational analysis result of NS_A2_B1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835179, upper bound: 1129.1835876
time: 1.14 seconds

## Relational analysis of NS_A2_B1_B1_A2_B2_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835179, upper bound: 1129.1835876
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -146.2314148, 561.0218506, -130.3862457, 502.5742493, -648.8056641, 691.4080811
1: -393.0219116, 1291.1850586, -351.0736694, 1158.4904785, -1551.5124512, 1642.2587891
2: -569.2935791, 1094.1822510, -505.2575989, 977.6444092, -1546.9379883, 1599.4398193
3: -332.6503296, 1377.9665527, -296.8209839, 1236.3240967, -1568.9743652, 1674.7874756
4: -524.7306519, 957.9172363, -465.8970642, 856.4816284, -1381.2122803, 1423.8140869

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_B1_A2_B2_B2_A1

### Relational analysis result of NS_A2_B1_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835179, upper bound: 1129.1835876
time: 0.87 seconds

## Relational analysis of NS_A2_B1_B1_A2_B2_B2_A2

### Relational analysis result of NS_A2_B1_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835179, upper bound: 1129.1835876
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -126.4143906, 486.3462524, -130.8279419, 503.5150757, -629.9293823, 617.1741943
1: -340.7070312, 1119.4364014, -351.8941040, 1159.6789551, -1500.3859863, 1471.3305664
2: -495.0114746, 943.7409668, -505.4384155, 981.9180298, -1476.9294434, 1449.1791992
3: -288.4903259, 1196.8377686, -297.3092346, 1237.0557861, -1525.5461426, 1494.1469727
4: -455.5537109, 826.9609375, -466.7855225, 859.6569214, -1315.2106934, 1293.7464600

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_B2_A1_A1_A1_B1

### Relational analysis result of NS_A2_B1_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1833883, upper bound: 1129.1836904
time: 0.92 seconds

## Relational analysis of NS_A2_B1_B2_A1_A1_A1_B2

### Relational analysis result of NS_A2_B1_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1833883, upper bound: 1129.1836990
time: 1.04 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -132.8479767, 511.4957581, -130.7947083, 503.3945312, -636.2424316, 642.2904663
1: -357.7890625, 1178.0196533, -351.8052979, 1159.4824219, -1517.2714844, 1529.8249512
2: -517.7007446, 993.9829102, -505.0921021, 981.7504272, -1499.4511719, 1499.0745850
3: -302.8157043, 1259.0712891, -297.2156067, 1236.8096924, -1539.6253662, 1556.2868652
4: -476.9688110, 870.9146729, -466.5547791, 859.5215454, -1336.4902344, 1337.4693604

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_B2_A1_A1_A2_B1

### Relational analysis result of NS_A2_B1_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1833889, upper bound: 1129.1837306
time: 1.22 seconds

## Relational analysis of NS_A2_B1_B2_A1_A1_A2_B2

### Relational analysis result of NS_A2_B1_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1833889, upper bound: 1129.1837692
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -138.0892792, 531.6274414, -131.2923889, 505.2796936, -643.3689575, 662.9197388
1: -371.9006653, 1223.8272705, -353.1367188, 1163.7135010, -1535.6141357, 1576.9639893
2: -537.4407959, 1033.5794678, -507.2873230, 985.2848511, -1522.7255859, 1540.8665771
3: -314.7073669, 1307.1173096, -298.3707581, 1241.4412842, -1556.1486816, 1605.4880371
4: -495.1140747, 905.4372559, -468.4702759, 862.6119385, -1357.7259521, 1373.9074707

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_B2_A1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1833772, upper bound: 1129.1833871
time: 0.99 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2_B1_A2

### Relational analysis result of NS_A2_B1_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1833772, upper bound: 1129.1836338
time: 1.05 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -138.0892792, 531.6274414, -138.9382782, 535.2903442, -673.3795166, 670.5657349
1: -371.9006653, 1223.8272705, -373.5462952, 1233.4838867, -1605.3845215, 1597.3735352
2: -537.4407959, 1033.5794678, -534.8917236, 1044.2347412, -1581.6755371, 1568.4711914
3: -314.7073669, 1307.1173096, -315.4972534, 1314.9692383, -1629.6766357, 1622.6145020
4: -495.1140747, 905.4372559, -494.4385376, 914.3750610, -1409.4891357, 1399.8757324

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_B2_A1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1833772, upper bound: 1129.1833871
time: 0.88 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1835238, upper bound: 1129.1836338
time: 1.41 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -146.8577118, 563.5684814, -131.1984406, 504.9515381, -651.8092651, 694.7668457
1: -394.8364258, 1296.5168457, -352.9292297, 1162.9967041, -1557.8331299, 1649.4460449
2: -571.9429932, 1098.8616943, -506.8584900, 984.7601318, -1556.7031250, 1605.7202148
3: -334.1492920, 1383.6822510, -298.1753845, 1240.5172119, -1574.6665039, 1681.8574219
4: -527.1762695, 962.0034180, -468.1056824, 862.1494751, -1389.3253174, 1430.1091309

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B2_A2_A1_B1_B1

### Relational analysis result of NS_A2_B1_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834444, upper bound: 1129.1836208
time: 0.98 seconds

## Relational analysis of NS_A2_B1_B2_A2_A1_B1_B2

### Relational analysis result of NS_A2_B1_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834444, upper bound: 1129.1835077
time: 1.01 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -146.8577118, 563.5684814, -135.8200226, 523.2105713, -670.0682983, 699.3884277
1: -394.8364258, 1296.5168457, -365.1741638, 1206.0606689, -1600.8970947, 1661.6910400
2: -571.9429932, 1098.8616943, -523.3812256, 1020.3242188, -1592.2672119, 1622.2429199
3: -334.1492920, 1383.6822510, -308.4322510, 1286.3287354, -1620.4780273, 1692.1145020
4: -527.1762695, 962.0034180, -483.7073669, 893.5848389, -1420.7611084, 1445.7108154

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B2_A2_A1_B2_B1

### Relational analysis result of NS_A2_B1_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834444, upper bound: 1129.1836255
time: 0.89 seconds

## Relational analysis of NS_A2_B1_B2_A2_A1_B2_B2

### Relational analysis result of NS_A2_B1_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834444, upper bound: 1129.1835091
time: 1.25 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -151.3416138, 581.2679443, -131.1984406, 504.9515381, -656.2930298, 712.4662476
1: -406.7064819, 1338.2929688, -352.9292297, 1162.9967041, -1569.7031250, 1691.2221680
2: -587.9353027, 1133.3332520, -506.8584900, 984.7601318, -1572.6954346, 1640.1916504
3: -344.1180115, 1428.1633301, -298.1753845, 1240.5172119, -1584.6352539, 1726.3385010
4: -542.2508545, 992.4601440, -468.1056824, 862.1494751, -1404.3995361, 1460.5657959

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B2_A2_A2_B1_B1

### Relational analysis result of NS_A2_B1_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834545, upper bound: 1129.1836770
time: 1.14 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834544, upper bound: 1129.1835639
time: 1.34 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -151.3416138, 581.2679443, -135.8200226, 523.2105713, -674.5520630, 717.0878906
1: -406.7064819, 1338.2929688, -365.1741638, 1206.0606689, -1612.7668457, 1703.4670410
2: -587.9353027, 1133.3332520, -523.3812256, 1020.3242188, -1608.2595215, 1656.7143555
3: -344.1180115, 1428.1633301, -308.4322510, 1286.3287354, -1630.4466553, 1736.5955811
4: -542.2508545, 992.4601440, -483.7073669, 893.5848389, -1435.8354492, 1476.1674805

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B2_A2_A2_B2_B1

### Relational analysis result of NS_A2_B1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834545, upper bound: 1129.1837034
time: 0.94 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2_B2_B2

### Relational analysis result of NS_A2_B1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834544, upper bound: 1129.1835877
time: 1.39 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -128.8547058, 495.6660156, -128.8547058, 495.6660156, -624.5206909, 624.5207520
1: -347.3490295, 1141.0483398, -347.3490295, 1141.0483398, -1488.3973389, 1488.3973389
2: -504.4868469, 961.9554443, -504.4868469, 961.9554443, -1466.4422607, 1466.4422607
3: -294.0944824, 1219.7583008, -294.0944824, 1219.7583008, -1513.8527832, 1513.8527832
4: -464.2849121, 842.9674683, -464.2849121, 842.9674683, -1307.2524414, 1307.2523193

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_B1_A1_A1_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1788451, upper bound: 1129.1790576
time: 1.03 seconds

## Relational analysis of NS_A2_B2_B1_A1_A1_B1_A2

### Relational analysis result of NS_A2_B2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1783796, upper bound: 1129.1788335
time: 1.07 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -128.8547058, 495.6660156, -135.1598663, 520.3241577, -649.1788330, 630.8258667
1: -347.3490295, 1141.0483398, -364.0769653, 1198.4763184, -1545.8251953, 1505.1252441
2: -504.4868469, 961.9554443, -526.6557007, 1011.2542114, -1515.7410889, 1488.6108398
3: -294.0944824, 1219.7583008, -308.1332703, 1280.7181396, -1574.8126221, 1527.8914795
4: -464.2849121, 842.9674683, -485.2278137, 886.0899048, -1350.3747559, 1328.1951904

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_B1_A1_A1_B2_B1

### Relational analysis result of NS_A2_B2_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1789086, upper bound: 1129.1792424
time: 0.91 seconds

## Relational analysis of NS_A2_B2_B1_A1_A1_B2_B2

### Relational analysis result of NS_A2_B2_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1783796, upper bound: 1129.1788335
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -135.1598663, 520.3241577, -128.8547058, 495.6660156, -630.8258057, 649.1788330
1: -364.0769653, 1198.4763184, -347.3490295, 1141.0483398, -1505.1252441, 1545.8251953
2: -526.6557007, 1011.2542114, -504.4868469, 961.9554443, -1488.6108398, 1515.7410889
3: -308.1332703, 1280.7181396, -294.0944824, 1219.7583008, -1527.8914795, 1574.8126221
4: -485.2278137, 886.0899048, -464.2849121, 842.9674683, -1328.1953125, 1350.3747559

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_B1_A1_A2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786062, upper bound: 1129.1786329
time: 0.93 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1778923, upper bound: 1129.1778923
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -135.1598663, 520.3241577, -135.1598663, 520.3241577, -655.4840088, 655.4840088
1: -364.0769653, 1198.4763184, -364.0769653, 1198.4763184, -1562.5532227, 1562.5532227
2: -526.6557007, 1011.2542114, -526.6557007, 1011.2542114, -1537.9096680, 1537.9096680
3: -308.1332703, 1280.7181396, -308.1332703, 1280.7181396, -1588.8514404, 1588.8514404
4: -485.2278137, 886.0899048, -485.2278137, 886.0899048, -1371.3177490, 1371.3177490

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_B1_A1_A2_B2_A1

### Relational analysis result of NS_A2_B2_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1786062, upper bound: 1129.1786329
time: 0.96 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2_B2_A2

### Relational analysis result of NS_A2_B2_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1778923, upper bound: 1129.1778923
time: 1.05 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -146.5412598, 562.3317871, -126.4143906, 486.3462524, -632.8875122, 688.7460938
1: -393.9497986, 1293.6365967, -340.7070312, 1119.4364014, -1513.3862305, 1634.3436279
2: -570.7597656, 1096.4041748, -495.0114746, 943.7409668, -1514.5003662, 1591.4156494
3: -333.4157104, 1380.6964111, -288.4903259, 1196.8377686, -1530.2534180, 1669.1866455
4: -526.0696411, 959.8532715, -455.5537109, 826.9609375, -1353.0305176, 1415.4069824

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_B1_A2_B1_B1_A1

### Relational analysis result of NS_A2_B2_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834378, upper bound: 1129.1832616
time: 1.38 seconds

## Relational analysis of NS_A2_B2_B1_A2_B1_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834378, upper bound: 1129.1834692
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -146.5236664, 562.2811890, -132.8479767, 511.4957581, -658.0193481, 695.1290894
1: -393.9027710, 1293.6445312, -357.7890625, 1178.0196533, -1571.9223633, 1651.4334717
2: -570.4169922, 1096.3686523, -517.7007446, 993.9829102, -1564.3999023, 1614.0693359
3: -333.3577881, 1380.6694336, -302.8157043, 1259.0712891, -1592.4290771, 1683.4851074
4: -525.8427124, 959.8358765, -476.9688110, 870.9146729, -1396.7573242, 1436.8044434

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_B1_A2_B1_B2_A1

### Relational analysis result of NS_A2_B2_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834623, upper bound: 1129.1832641
time: 1.42 seconds

## Relational analysis of NS_A2_B2_B1_A2_B1_B2_A2

### Relational analysis result of NS_A2_B2_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834623, upper bound: 1129.1834958
time: 1.19 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -146.9880524, 564.0287476, -138.0892792, 531.6274414, -678.6153564, 702.1179199
1: -395.1430359, 1297.5109863, -371.9006653, 1223.8272705, -1618.9703369, 1669.4116211
2: -572.5503540, 1099.6473389, -537.4407959, 1033.5794678, -1606.1297607, 1637.0881348
3: -334.4413452, 1384.9193115, -314.7073669, 1307.1173096, -1641.5585938, 1699.6267090
4: -527.7011108, 962.6979370, -495.1140747, 905.4372559, -1433.1384277, 1457.8120117

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834168, upper bound: 1129.1834918
time: 1.10 seconds

## Relational analysis of NS_A2_B2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1831495, upper bound: 1129.1828885
time: 0.99 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -153.8633423, 591.0927734, -138.0892792, 531.6274414, -685.4907837, 729.1819458
1: -413.3982849, 1360.4058838, -371.9006653, 1223.8272705, -1637.2255859, 1732.3065186
2: -597.2430420, 1152.7336426, -537.4407959, 1033.5794678, -1630.8225098, 1690.1744385
3: -349.7688293, 1451.2940674, -314.7073669, 1307.1173096, -1656.8861084, 1766.0014648
4: -550.8772583, 1009.3328857, -495.1140747, 905.4372559, -1456.3144531, 1504.4470215

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1831640, upper bound: 1129.1833119
time: 0.96 seconds

## Relational analysis of NS_A2_B2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834154, upper bound: 1129.1834640
time: 0.99 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -126.4143906, 486.3462524, -146.5412598, 562.3317871, -688.7460938, 632.8875122
1: -340.7070312, 1119.4364014, -393.9497986, 1293.6365967, -1634.3436279, 1513.3862305
2: -495.0114746, 943.7409668, -570.7597656, 1096.4041748, -1591.4156494, 1514.5004883
3: -288.4903259, 1196.8377686, -333.4157104, 1380.6964111, -1669.1866455, 1530.2534180
4: -455.5537109, 826.9609375, -526.0696411, 959.8532715, -1415.4069824, 1353.0305176

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_B2_A1_A1_A1_B1

### Relational analysis result of NS_A2_B2_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1832616, upper bound: 1129.1834378
time: 0.88 seconds

## Relational analysis of NS_A2_B2_B2_A1_A1_A1_B2

### Relational analysis result of NS_A2_B2_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1832616, upper bound: 1129.1834844
time: 1.16 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -132.8479767, 511.4957581, -146.5236664, 562.2811890, -695.1290894, 658.0193481
1: -357.7890625, 1178.0196533, -393.9027710, 1293.6445312, -1651.4334717, 1571.9223633
2: -517.7007446, 993.9829102, -570.4169922, 1096.3686523, -1614.0693359, 1564.3999023
3: -302.8157043, 1259.0712891, -333.3577881, 1380.6694336, -1683.4851074, 1592.4290771
4: -476.9688110, 870.9146729, -525.8427124, 959.8358765, -1436.8044434, 1396.7573242

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_B2_A1_A1_A2_B1

### Relational analysis result of NS_A2_B2_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1832641, upper bound: 1129.1834623
time: 0.93 seconds

## Relational analysis of NS_A2_B2_B2_A1_A1_A2_B2

### Relational analysis result of NS_A2_B2_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1832641, upper bound: 1129.1835743
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -138.0892792, 531.6274414, -146.9880524, 564.0287476, -702.1179199, 678.6153564
1: -371.9006653, 1223.8272705, -395.1430359, 1297.5109863, -1669.4116211, 1618.9703369
2: -537.4407959, 1033.5794678, -572.5503540, 1099.6473389, -1637.0881348, 1606.1298828
3: -314.7073669, 1307.1173096, -334.4413452, 1384.9193115, -1699.6267090, 1641.5585938
4: -495.1140747, 905.4372559, -527.7011108, 962.6979370, -1457.8120117, 1433.1384277

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_B2_A1_A2_B1_A1

### Relational analysis result of NS_A2_B2_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834918, upper bound: 1129.1834168
time: 1.32 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2_B1_A2

### Relational analysis result of NS_A2_B2_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1828885, upper bound: 1129.1831495
time: 1.12 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -138.0892792, 531.6274414, -153.8633423, 591.0927734, -729.1819458, 685.4907837
1: -371.9006653, 1223.8272705, -413.3982849, 1360.4058838, -1732.3065186, 1637.2255859
2: -537.4407959, 1033.5794678, -597.2430420, 1152.7336426, -1690.1744385, 1630.8225098
3: -314.7073669, 1307.1173096, -349.7688293, 1451.2940674, -1766.0014648, 1656.8861084
4: -495.1140747, 905.4372559, -550.8772583, 1009.3328857, -1504.4470215, 1456.3144531

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1833119, upper bound: 1129.1831640
time: 1.44 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1833119, upper bound: 1129.1834154
time: 1.01 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -146.8577118, 563.5684814, -146.8577118, 563.5684814, -710.4261475, 710.4261475
1: -394.8364258, 1296.5168457, -394.8364258, 1296.5168457, -1691.3532715, 1691.3532715
2: -571.9429932, 1098.8616943, -571.9429932, 1098.8616943, -1670.8046875, 1670.8046875
3: -334.1492920, 1383.6822510, -334.1492920, 1383.6822510, -1717.8315430, 1717.8315430
4: -527.1762695, 962.0034180, -527.1762695, 962.0034180, -1489.1796875, 1489.1796875

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1832382, upper bound: 1129.1833195
time: 1.01 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1833141, upper bound: 1129.1833141
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -151.3416138, 581.2679443, -146.8577118, 563.5684814, -714.9100342, 728.1256104
1: -406.7064819, 1338.2929688, -394.8364258, 1296.5168457, -1703.2233887, 1733.1293945
2: -587.9353027, 1133.3332520, -571.9429932, 1098.8616943, -1686.7969971, 1705.2758789
3: -344.1180115, 1428.1633301, -334.1492920, 1383.6822510, -1727.7999268, 1762.3126221
4: -542.2508545, 992.4601440, -527.1762695, 962.0034180, -1504.2541504, 1519.6364746

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_B2_A2_B1_A2_A1

### Relational analysis result of NS_A2_B2_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1834226, upper bound: 1129.1834527
time: 1.22 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_A2_A2

### Relational analysis result of NS_A2_B2_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1833970, upper bound: 1129.1834526
time: 0.96 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -146.8577118, 563.5684814, -151.3416138, 581.2679443, -728.1256104, 714.9100342
1: -394.8364258, 1296.5168457, -406.7064819, 1338.2929688, -1733.1293945, 1703.2233887
2: -571.9429932, 1098.8616943, -587.9353027, 1133.3332520, -1705.2758789, 1686.7969971
3: -334.1492920, 1383.6822510, -344.1180115, 1428.1633301, -1762.3126221, 1727.8001709
4: -527.1762695, 962.0034180, -542.2508545, 992.4601440, -1519.6364746, 1504.2541504

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1833795, upper bound: 1129.1834266
time: 0.93 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1833970, upper bound: 1129.1834009
time: 1.36 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -151.3416138, 581.2679443, -151.3416138, 581.2679443, -732.6094360, 732.6094360
1: -406.7064819, 1338.2929688, -406.7064819, 1338.2929688, -1744.9993896, 1744.9993896
2: -587.9353027, 1133.3332520, -587.9353027, 1133.3332520, -1721.2683105, 1721.2683105
3: -344.1180115, 1428.1633301, -344.1180115, 1428.1633301, -1772.2811279, 1772.2811279
4: -542.2508545, 992.4601440, -542.2508545, 992.4601440, -1534.7109375, 1534.7109375

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_B2_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1833155, upper bound: 1129.1833769
time: 0.99 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1129.1833141, upper bound: 1129.1834386
time: 1.12 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 4.01 seconds
NS_A1_B1_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1838620, upper bound: 1129.1838834
NS_A1_B1_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1838490, upper bound: 1129.1838513
NS_A1_B1_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1838671, upper bound: 1129.1838570
NS_A1_B1_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1838490, upper bound: 1129.1838513
NS_A1_B1_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1838454, upper bound: 1129.1838818
NS_A1_B1_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1838374, upper bound: 1129.1838490
NS_A1_B1_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1838454, upper bound: 1129.1838818
NS_A1_B1_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1838374, upper bound: 1129.1838490
NS_A1_B1_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1835607, upper bound: 1129.1835146
NS_A1_B1_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1835607, upper bound: 1129.1835146
NS_A1_B1_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1835607, upper bound: 1129.1835146
NS_A1_B1_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1835607, upper bound: 1129.1835146
NS_A1_B1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1835708, upper bound: 1129.1835833
NS_A1_B1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1834346, upper bound: 1129.1834777
NS_A1_B1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1835708, upper bound: 1129.1835833
NS_A1_B1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1834346, upper bound: 1129.1834777
NS_A1_B1_B2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1835146, upper bound: 1129.1835607
NS_A1_B1_B2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1835146, upper bound: 1129.1835607
NS_A1_B1_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1835146, upper bound: 1129.1835607
NS_A1_B1_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1835146, upper bound: 1129.1835607
NS_A1_B1_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1835833, upper bound: 1129.1835708
NS_A1_B1_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1834777, upper bound: 1129.1834346
NS_A1_B1_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1835833, upper bound: 1129.1835708
NS_A1_B1_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1834777, upper bound: 1129.1834346
NS_A1_B1_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1837227, upper bound: 1129.1837035
NS_A1_B1_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1836710, upper bound: 1129.1836852
NS_A1_B1_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1837227, upper bound: 1129.1837035
NS_A1_B1_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1836710, upper bound: 1129.1836852
NS_A1_B1_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1836803, upper bound: 1129.1837463
NS_A1_B1_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1836710, upper bound: 1129.1837002
NS_A1_B1_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1836803, upper bound: 1129.1837476
NS_A1_B1_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1836710, upper bound: 1129.1837002
NS_A1_B2_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1838400, upper bound: 1129.1836939
NS_A1_B2_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1837450, upper bound: 1129.1836939
NS_A1_B2_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1838400, upper bound: 1129.1837087
NS_A1_B2_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1837450, upper bound: 1129.1837087
NS_A1_B2_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1838355, upper bound: 1129.1836808
NS_A1_B2_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1837450, upper bound: 1129.1836808
NS_A1_B2_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1838355, upper bound: 1129.1837023
NS_A1_B2_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1837450, upper bound: 1129.1837023
NS_A1_B2_A1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1836799, upper bound: 1129.1836397
NS_A1_B2_A1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1836754, upper bound: 1129.1836577
NS_A1_B2_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1837122, upper bound: 1129.1835523
NS_A1_B2_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1837122, upper bound: 1129.1835556
NS_A1_B2_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1835876, upper bound: 1129.1835179
NS_A1_B2_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1835876, upper bound: 1129.1835179
NS_A1_B2_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1835876, upper bound: 1129.1835179
NS_A1_B2_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1835876, upper bound: 1129.1835179
NS_A1_B2_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1836904, upper bound: 1129.1833883
NS_A1_B2_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1836904, upper bound: 1129.1835363
NS_A1_B2_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1837306, upper bound: 1129.1833889
NS_A1_B2_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1837306, upper bound: 1129.1835531
NS_A1_B2_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1833871, upper bound: 1129.1833772
NS_A1_B2_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1836338, upper bound: 1129.1835238
NS_A1_B2_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1833871, upper bound: 1129.1833772
NS_A1_B2_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1836338, upper bound: 1129.1835238
NS_A1_B2_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1836160, upper bound: 1129.1834444
NS_A1_B2_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1835075, upper bound: 1129.1834444
NS_A1_B2_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1836160, upper bound: 1129.1834912
NS_A1_B2_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1835075, upper bound: 1129.1834912
NS_A1_B2_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1836160, upper bound: 1129.1834545
NS_A1_B2_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1835075, upper bound: 1129.1834544
NS_A1_B2_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1836160, upper bound: 1129.1835234
NS_A1_B2_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1835075, upper bound: 1129.1835234
NS_A2_B1_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1836939, upper bound: 1129.1838400
NS_A2_B1_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1836939, upper bound: 1129.1837450
NS_A2_B1_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1836939, upper bound: 1129.1838400
NS_A2_B1_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1836939, upper bound: 1129.1837450
NS_A2_B1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1836808, upper bound: 1129.1838499
NS_A2_B1_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1836808, upper bound: 1129.1837710
NS_A2_B1_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1836808, upper bound: 1129.1838499
NS_A2_B1_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1836808, upper bound: 1129.1837710
NS_A2_B1_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1836397, upper bound: 1129.1836799
NS_A2_B1_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1836577, upper bound: 1129.1836754
NS_A2_B1_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1835523, upper bound: 1129.1837122
NS_A2_B1_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1835523, upper bound: 1129.1837122
NS_A2_B1_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1835179, upper bound: 1129.1835876
NS_A2_B1_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1835179, upper bound: 1129.1835876
NS_A2_B1_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1835179, upper bound: 1129.1835876
NS_A2_B1_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1835179, upper bound: 1129.1835876
NS_A2_B1_B2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1833883, upper bound: 1129.1836904
NS_A2_B1_B2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1833883, upper bound: 1129.1836990
NS_A2_B1_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1833889, upper bound: 1129.1837306
NS_A2_B1_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1833889, upper bound: 1129.1837692
NS_A2_B1_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1833772, upper bound: 1129.1833871
NS_A2_B1_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1833772, upper bound: 1129.1836338
NS_A2_B1_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1833772, upper bound: 1129.1833871
NS_A2_B1_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1835238, upper bound: 1129.1836338
NS_A2_B1_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1834444, upper bound: 1129.1836208
NS_A2_B1_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1834444, upper bound: 1129.1835077
NS_A2_B1_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1834444, upper bound: 1129.1836255
NS_A2_B1_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1834444, upper bound: 1129.1835091
NS_A2_B1_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1834545, upper bound: 1129.1836770
NS_A2_B1_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1834544, upper bound: 1129.1835639
NS_A2_B1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1834545, upper bound: 1129.1837034
NS_A2_B1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1834544, upper bound: 1129.1835877
NS_A2_B2_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1788451, upper bound: 1129.1790576
NS_A2_B2_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1783796, upper bound: 1129.1788335
NS_A2_B2_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1789086, upper bound: 1129.1792424
NS_A2_B2_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1783796, upper bound: 1129.1788335
NS_A2_B2_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1786062, upper bound: 1129.1786329
NS_A2_B2_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1778923, upper bound: 1129.1778923
NS_A2_B2_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1786062, upper bound: 1129.1786329
NS_A2_B2_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1778923, upper bound: 1129.1778923
NS_A2_B2_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1834378, upper bound: 1129.1832616
NS_A2_B2_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1834378, upper bound: 1129.1834692
NS_A2_B2_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1834623, upper bound: 1129.1832641
NS_A2_B2_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1834623, upper bound: 1129.1834958
NS_A2_B2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1834168, upper bound: 1129.1834918
NS_A2_B2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1831495, upper bound: 1129.1828885
NS_A2_B2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1831640, upper bound: 1129.1833119
NS_A2_B2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1834154, upper bound: 1129.1834640
NS_A2_B2_B2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1832616, upper bound: 1129.1834378
NS_A2_B2_B2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1832616, upper bound: 1129.1834844
NS_A2_B2_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1832641, upper bound: 1129.1834623
NS_A2_B2_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1832641, upper bound: 1129.1835743
NS_A2_B2_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1834918, upper bound: 1129.1834168
NS_A2_B2_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1828885, upper bound: 1129.1831495
NS_A2_B2_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1833119, upper bound: 1129.1831640
NS_A2_B2_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1833119, upper bound: 1129.1834154
NS_A2_B2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1832382, upper bound: 1129.1833195
NS_A2_B2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1833141, upper bound: 1129.1833141
NS_A2_B2_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1834226, upper bound: 1129.1834527
NS_A2_B2_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1833970, upper bound: 1129.1834526
NS_A2_B2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1833795, upper bound: 1129.1834266
NS_A2_B2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1833970, upper bound: 1129.1834009
NS_A2_B2_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1833155, upper bound: 1129.1833769
NS_A2_B2_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 3, lower bound: -1129.1833141, upper bound: 1129.1834386

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.40 + 417.49 = 420.88 seconds
