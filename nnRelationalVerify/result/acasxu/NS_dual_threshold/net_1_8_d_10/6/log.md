## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 6)
Time budget: 420 seconds
Split limit: 100
Threshold: 1339.202386442214


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-260.5380249, 235.5966949, -260.5380249, 235.5966949, -496.1347046, 496.1347046)
1: (-994.5258789, 850.3129272, -994.5258789, 850.3129272, -1844.8388672, 1844.8388672)
2: (-529.1870117, 881.2379150, -529.1870117, 881.2379150, -1410.4249268, 1410.4249268)
3: (-910.2102051, 777.3389282, -910.2102051, 777.3389282, -1687.5490723, 1687.5490723)
4: (-664.5346680, 894.3054810, -664.5346680, 894.3054810, -1558.8400879, 1558.8399658)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.18 + 2.29 = 3.47 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -1339.2157786, upper bound: 1339.2157786

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2147286, upper bound: 1339.2157786
time: 0.92 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2147286, upper bound: 1339.2147286
time: 0.91 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.93 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 1.93
Output dim: 3, lower bound: -1339.2147286, upper bound: 1339.2157786
NS_B2, status: Status.UNKNOWN, split count: 1, time: 1.93
Output dim: 3, lower bound: -1339.2147286, upper bound: 1339.2147286

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -256.1399231, 231.4644623, -253.9733124, 229.4320374, -485.5719604, 485.4377441
1: -978.1137085, 835.6396484, -970.0575562, 828.4106445, -1806.5242920, 1805.6972656
2: -519.8786011, 865.7119141, -515.2947388, 858.0770874, -1377.9555664, 1381.0065918
3: -894.7660522, 764.0308838, -887.2174072, 757.4891968, -1652.2552490, 1651.2482910
4: -653.0750122, 878.5214844, -647.4416504, 870.7501831, -1523.8251953, 1525.9630127

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 45

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2147286, upper bound: 1339.2147286
time: 0.93 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2147286, upper bound: 1339.2147286
time: 1.02 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -257.2024536, 232.5796967, -272.3238220, 245.8114471, -503.0138245, 504.9035034
1: -981.9048462, 839.5596924, -1040.8454590, 887.8678589, -1869.7724609, 1880.4051514
2: -522.2640381, 869.8874512, -552.4603271, 919.3123779, -1441.5761719, 1422.3477783
3: -898.4796143, 767.5150146, -951.5145874, 811.9313965, -1710.4108887, 1719.0294189
4: -655.9251099, 882.8465576, -694.0728760, 933.0305786, -1588.9556885, 1576.9193115

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2147286, upper bound: 1339.2147286
time: 0.90 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2147286, upper bound: 1339.2147286
time: 1.07 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.16 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 3.16
Output dim: 3, lower bound: -1339.2147286, upper bound: 1339.2147286
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 3.16
Output dim: 3, lower bound: -1339.2147286, upper bound: 1339.2147286
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 3.16
Output dim: 3, lower bound: -1339.2147286, upper bound: 1339.2147286
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 3.16
Output dim: 3, lower bound: -1339.2147286, upper bound: 1339.2147286

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -253.9733124, 229.4320374, -253.9733124, 229.4320374, -483.4052429, 483.4052429
1: -970.0575562, 828.4106445, -970.0575562, 828.4106445, -1798.4681396, 1798.4681396
2: -515.2947388, 858.0770874, -515.2947388, 858.0770874, -1373.3718262, 1373.3718262
3: -887.2174072, 757.4891968, -887.2174072, 757.4891968, -1644.7065430, 1644.7065430
4: -647.4416504, 870.7501831, -647.4416504, 870.7501831, -1518.1918945, 1518.1918945

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B1_A1_B1

### Relational analysis result of NS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2103701, upper bound: 1339.2144566
time: 1.12 seconds

## Relational analysis of NS_B1_A1_B2

### Relational analysis result of NS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2102833, upper bound: 1339.2128235
time: 1.01 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -272.3238220, 245.8114471, -253.9733124, 229.4320374, -501.7558289, 499.7846375
1: -1040.8454590, 887.8678589, -970.0575562, 828.4106445, -1869.2561035, 1857.9250488
2: -552.4603271, 919.3123779, -515.2947388, 858.0770874, -1410.5373535, 1434.6071777
3: -951.5145874, 811.9313965, -887.2174072, 757.4891968, -1709.0036621, 1699.1488037
4: -694.0728760, 933.0305786, -647.4416504, 870.7501831, -1564.8229980, 1580.4721680

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2147286, upper bound: 1339.2156194
time: 0.95 seconds

## Relational analysis of NS_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2145492, upper bound: 1339.2156194
time: 1.47 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -253.9733124, 229.4320374, -272.3238220, 245.8114471, -499.7846375, 501.7558289
1: -970.0575562, 828.4106445, -1040.8454590, 887.8678589, -1857.9250488, 1869.2561035
2: -515.2947388, 858.0770874, -552.4603271, 919.3123779, -1434.6071777, 1410.5373535
3: -887.2174072, 757.4891968, -951.5145874, 811.9313965, -1699.1488037, 1709.0037842
4: -647.4416504, 870.7501831, -694.0728760, 933.0305786, -1580.4721680, 1564.8229980

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_A1

### Relational analysis result of NS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2145492, upper bound: 1339.2147286
time: 1.06 seconds

## Relational analysis of NS_B2_A1_A2

### Relational analysis result of NS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2145492, upper bound: 1339.2145492
time: 0.97 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -272.3238220, 245.8114471, -272.3238220, 245.8114471, -518.1352539, 518.1352539
1: -1040.8454590, 887.8678589, -1040.8454590, 887.8678589, -1928.7131348, 1928.7131348
2: -552.4603271, 919.3123779, -552.4603271, 919.3123779, -1471.7727051, 1471.7727051
3: -951.5145874, 811.9313965, -951.5145874, 811.9313965, -1763.4456787, 1763.4456787
4: -694.0728760, 933.0305786, -694.0728760, 933.0305786, -1627.1033936, 1627.1035156

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B2_A2_A1

### Relational analysis result of NS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2140579, upper bound: 1339.2103701
time: 1.14 seconds

## Relational analysis of NS_B2_A2_A2

### Relational analysis result of NS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2102833, upper bound: 1339.2102833
time: 0.96 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.44 seconds
NS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.44
Output dim: 3, lower bound: -1339.2103701, upper bound: 1339.2144566
NS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.44
Output dim: 3, lower bound: -1339.2102833, upper bound: 1339.2128235
NS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.44
Output dim: 3, lower bound: -1339.2147286, upper bound: 1339.2156194
NS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.44
Output dim: 3, lower bound: -1339.2145492, upper bound: 1339.2156194
NS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 3.44
Output dim: 3, lower bound: -1339.2145492, upper bound: 1339.2147286
NS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 3.44
Output dim: 3, lower bound: -1339.2145492, upper bound: 1339.2145492
NS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 3.44
Output dim: 3, lower bound: -1339.2140579, upper bound: 1339.2103701
NS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.44
Output dim: 3, lower bound: -1339.2102833, upper bound: 1339.2102833

## BFS NS instance: NS_B1_A1_B1

### Backsubstitution after applying NS history:
0: -251.5078888, 227.1967926, -250.2987823, 226.1052246, -477.6130676, 477.4955750
1: -960.6152344, 820.1411743, -955.9870605, 816.1038818, -1776.7191162, 1776.1281738
2: -510.3280945, 849.9970703, -507.8896484, 846.0427246, -1356.3708496, 1357.8867188
3: -878.6581421, 750.0983276, -874.4613037, 746.4773560, -1625.1354980, 1624.5593262
4: -641.1159058, 862.4911499, -638.0168457, 858.4481812, -1499.5639648, 1500.5080566

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_A1_B1_A1

### Relational analysis result of NS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2128548, upper bound: 1339.2128548
time: 0.88 seconds

## Relational analysis of NS_B1_A1_B1_A2

### Relational analysis result of NS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2128548, upper bound: 1339.2128548
time: 1.08 seconds

## BFS NS instance: NS_B1_A1_B2

### Backsubstitution after applying NS history:
0: -252.0396271, 227.7207031, -258.1776123, 233.3300476, -485.3696594, 485.8983154
1: -962.5991821, 822.1835327, -986.4415894, 842.1447754, -1804.7438965, 1808.6247559
2: -511.4775696, 851.8244629, -524.4260864, 873.4448242, -1384.9222412, 1376.2498779
3: -880.4241943, 751.8280640, -901.9034424, 770.3521729, -1650.7763672, 1653.7314453
4: -642.4992065, 864.4022217, -658.0135498, 886.3843994, -1528.8835449, 1522.4157715

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_A1_B2_A1

### Relational analysis result of NS_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2128548, upper bound: 1339.2128548
time: 0.94 seconds

## Relational analysis of NS_B1_A1_B2_A2

### Relational analysis result of NS_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2128548, upper bound: 1339.2128548
time: 1.01 seconds

## BFS NS instance: NS_B1_A2_B1

### Backsubstitution after applying NS history:
0: -271.2564087, 244.8452759, -249.8258362, 225.6879120, -496.9443359, 494.6711121
1: -1036.7397461, 884.3836670, -954.0773315, 814.9205322, -1851.6602783, 1838.4606934
2: -550.3150024, 915.7041016, -506.9686279, 844.0984497, -1394.4132080, 1422.6726074
3: -947.8121948, 808.7298584, -872.8045044, 745.0787964, -1692.8909912, 1681.5344238
4: -691.3849487, 929.3829956, -636.9813843, 856.6110840, -1547.9959717, 1566.3643799

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A2_B1_B1

### Relational analysis result of NS_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2138730, upper bound: 1339.2133463
time: 1.02 seconds

## Relational analysis of NS_B1_A2_B1_B2

### Relational analysis result of NS_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2138820, upper bound: 1339.2145348
time: 1.02 seconds

## BFS NS instance: NS_B1_A2_B2

### Backsubstitution after applying NS history:
0: -271.3485413, 244.9192505, -316.2953796, 283.6633301, -555.0118408, 561.2145996
1: -1037.1108398, 884.6646729, -1211.0053711, 1024.1472168, -2061.2578125, 2095.6699219
2: -550.4669800, 916.0018311, -639.8353882, 1062.9650879, -1613.4321289, 1555.8371582
3: -948.1159058, 808.9785767, -1104.7574463, 938.4837646, -1886.5996094, 1913.7359619
4: -691.5877075, 929.6611328, -804.7814941, 1076.7966309, -1768.3842773, 1734.4425049

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_A2_B2_A1

### Relational analysis result of NS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2145492, upper bound: 1339.2156194
time: 0.90 seconds

## Relational analysis of NS_B1_A2_B2_A2

### Relational analysis result of NS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2145492, upper bound: 1339.2156194
time: 0.99 seconds

## BFS NS instance: NS_B2_A1_A1

### Backsubstitution after applying NS history:
0: -249.8258362, 225.6879120, -271.2564087, 244.8452759, -494.6711121, 496.9443359
1: -954.0773315, 814.9205322, -1036.7397461, 884.3836670, -1838.4606934, 1851.6602783
2: -506.9686279, 844.0984497, -550.3150024, 915.7041016, -1422.6726074, 1394.4132080
3: -872.8045044, 745.0787964, -947.8121948, 808.7298584, -1681.5344238, 1692.8909912
4: -636.9813843, 856.6110840, -691.3849487, 929.3829956, -1566.3643799, 1547.9958496

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_A1_A1_A1

### Relational analysis result of NS_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2133463, upper bound: 1339.2138730
time: 1.08 seconds

## Relational analysis of NS_B2_A1_A1_A2

### Relational analysis result of NS_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2145348, upper bound: 1339.2138820
time: 0.89 seconds

## BFS NS instance: NS_B2_A1_A2

### Backsubstitution after applying NS history:
0: -316.2953796, 283.6633301, -271.3485413, 244.9192505, -561.2145996, 555.0118408
1: -1211.0053711, 1024.1472168, -1037.1108398, 884.6646729, -2095.6699219, 2061.2575684
2: -639.8353882, 1062.9650879, -550.4669800, 916.0018311, -1555.8371582, 1613.4321289
3: -1104.7574463, 938.4837646, -948.1159058, 808.9785767, -1913.7359619, 1886.5996094
4: -804.7814941, 1076.7966309, -691.5877075, 929.6611328, -1734.4425049, 1768.3842773

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_A2_B1

### Relational analysis result of NS_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2156194, upper bound: 1339.2145492
time: 1.42 seconds

## Relational analysis of NS_B2_A1_A2_B2

### Relational analysis result of NS_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2156194, upper bound: 1339.2145492
time: 0.95 seconds

## BFS NS instance: NS_B2_A2_A1

### Backsubstitution after applying NS history:
0: -268.3889771, 242.2476959, -269.7073364, 243.4367371, -511.8256531, 511.9550171
1: -1025.7521973, 874.7044067, -1030.8115234, 879.0928345, -1904.8443604, 1905.5157471
2: -544.5684204, 906.4784546, -547.2086182, 910.7730713, -1455.3414307, 1453.6867676
3: -937.8247070, 800.2047729, -942.4113159, 804.1304932, -1741.9552002, 1742.6159668
4: -683.9780273, 919.9148560, -687.3574219, 924.3040771, -1608.2821045, 1607.2722168

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A2_A1_B1

### Relational analysis result of NS_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2102833, upper bound: 1339.2102833
time: 0.94 seconds

## Relational analysis of NS_B2_A2_A1_B2

### Relational analysis result of NS_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2102833, upper bound: 1339.2102833
time: 0.97 seconds

## BFS NS instance: NS_B2_A2_A2

### Backsubstitution after applying NS history:
0: -273.1560669, 246.9745636, -270.3593750, 244.0794983, -517.2354736, 517.3339233
1: -1044.1036377, 891.4555054, -1033.2656250, 881.5689087, -1925.6726074, 1924.7211914
2: -555.3973389, 923.8026733, -548.5484619, 912.9200439, -1468.3173828, 1472.3507080
3: -954.4480591, 815.1679077, -944.6306152, 806.1697998, -1760.6179199, 1759.7983398
4: -696.1833496, 937.7105713, -689.0526733, 926.5512085, -1622.7346191, 1626.7631836

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A2_A2_B1

### Relational analysis result of NS_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2102833, upper bound: 1339.2102833
time: 0.98 seconds

## Relational analysis of NS_B2_A2_A2_B2

### Relational analysis result of NS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2102833, upper bound: 1339.2102833
time: 0.90 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.98 seconds
NS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 3, lower bound: -1339.2128548, upper bound: 1339.2128548
NS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 3, lower bound: -1339.2128548, upper bound: 1339.2128548
NS_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 3, lower bound: -1339.2128548, upper bound: 1339.2128548
NS_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 3, lower bound: -1339.2128548, upper bound: 1339.2128548
NS_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 3, lower bound: -1339.2138730, upper bound: 1339.2133463
NS_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 3, lower bound: -1339.2138820, upper bound: 1339.2145348
NS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 3, lower bound: -1339.2145492, upper bound: 1339.2156194
NS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 3, lower bound: -1339.2145492, upper bound: 1339.2156194
NS_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 3, lower bound: -1339.2133463, upper bound: 1339.2138730
NS_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 3, lower bound: -1339.2145348, upper bound: 1339.2138820
NS_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 3, lower bound: -1339.2156194, upper bound: 1339.2145492
NS_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 3, lower bound: -1339.2156194, upper bound: 1339.2145492
NS_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 3, lower bound: -1339.2102833, upper bound: 1339.2102833
NS_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 3, lower bound: -1339.2102833, upper bound: 1339.2102833
NS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 3, lower bound: -1339.2102833, upper bound: 1339.2102833
NS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 3, lower bound: -1339.2102833, upper bound: 1339.2102833

## BFS NS instance: NS_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -250.2987823, 226.1052246, -250.2987823, 226.1052246, -476.4039307, 476.4039307
1: -955.9870605, 816.1038818, -955.9870605, 816.1038818, -1772.0909424, 1772.0909424
2: -507.8896484, 846.0427246, -507.8896484, 846.0427246, -1353.9322510, 1353.9322510
3: -874.4613037, 746.4773560, -874.4613037, 746.4773560, -1620.9383545, 1620.9383545
4: -638.0168457, 858.4481812, -638.0168457, 858.4481812, -1496.4649658, 1496.4650879

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_B1_A1_A1

### Relational analysis result of NS_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2102508, upper bound: 1339.2123506
time: 0.94 seconds

## Relational analysis of NS_B1_A1_B1_A1_A2

### Relational analysis result of NS_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2123150, upper bound: 1339.2138753
time: 1.32 seconds

## BFS NS instance: NS_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -258.1776123, 233.3300476, -250.2987823, 226.1052246, -484.2828369, 483.6287842
1: -986.4415894, 842.1447754, -955.9870605, 816.1038818, -1802.5449219, 1798.1318359
2: -524.4260864, 873.4448242, -507.8896484, 846.0427246, -1370.4685059, 1381.3343506
3: -901.9034424, 770.3521729, -874.4613037, 746.4773560, -1648.3808594, 1644.8134766
4: -658.0135498, 886.3843994, -638.0168457, 858.4481812, -1516.4616699, 1524.4012451

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_B1_A2_B1

### Relational analysis result of NS_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2129796, upper bound: 1339.2143105
time: 1.02 seconds

## Relational analysis of NS_B1_A1_B1_A2_B2

### Relational analysis result of NS_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2127438, upper bound: 1339.2143105
time: 1.03 seconds

## BFS NS instance: NS_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -250.2987823, 226.1052246, -258.1776123, 233.3300476, -483.6287842, 484.2828369
1: -955.9870605, 816.1038818, -986.4415894, 842.1447754, -1798.1318359, 1802.5449219
2: -507.8896484, 846.0427246, -524.4260864, 873.4448242, -1381.3343506, 1370.4685059
3: -874.4613037, 746.4773560, -901.9034424, 770.3521729, -1644.8134766, 1648.3808594
4: -638.0168457, 858.4481812, -658.0135498, 886.3843994, -1524.4012451, 1516.4616699

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_B2_A1_A1

### Relational analysis result of NS_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2125363, upper bound: 1339.2127501
time: 0.93 seconds

## Relational analysis of NS_B1_A1_B2_A1_A2

### Relational analysis result of NS_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2124666, upper bound: 1339.2124666
time: 0.99 seconds

## BFS NS instance: NS_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -258.1776123, 233.3300476, -258.1776123, 233.3300476, -491.5076599, 491.5076599
1: -986.4415894, 842.1447754, -986.4415894, 842.1447754, -1828.5861816, 1828.5861816
2: -524.4260864, 873.4448242, -524.4260864, 873.4448242, -1397.8706055, 1397.8706055
3: -901.9034424, 770.3521729, -901.9034424, 770.3521729, -1672.2556152, 1672.2556152
4: -658.0135498, 886.3843994, -658.0135498, 886.3843994, -1544.3979492, 1544.3979492

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_B2_A2_B1

### Relational analysis result of NS_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2116455, upper bound: 1339.2100258
time: 0.99 seconds

## Relational analysis of NS_B1_A1_B2_A2_B2

### Relational analysis result of NS_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2119652, upper bound: 1339.2119652
time: 1.01 seconds

## BFS NS instance: NS_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -265.3952332, 239.5085449, -240.2304993, 216.8313446, -482.2265625, 479.7390442
1: -1014.6293335, 864.8557739, -917.9259644, 782.8681030, -1797.4974365, 1782.7816162
2: -539.0120239, 895.3712769, -487.6299133, 810.6806030, -1349.6925049, 1383.0010986
3: -927.1965942, 790.8286743, -839.0195312, 715.6842041, -1642.8808594, 1629.8480225
4: -676.4360962, 908.6753540, -612.4997559, 822.3607178, -1498.7968750, 1521.1749268

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A2_B1_B1_A1

### Relational analysis result of NS_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2128453, upper bound: 1339.2132609
time: 1.01 seconds

## Relational analysis of NS_B1_A2_B1_B1_A2

### Relational analysis result of NS_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2128453, upper bound: 1339.2132609
time: 0.95 seconds

## BFS NS instance: NS_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -265.7434998, 239.8792419, -271.8391724, 245.5500793, -511.2935791, 511.7183838
1: -1015.7272949, 866.4550171, -1040.1838379, 887.8743286, -1903.6015625, 1906.6387939
2: -539.1776733, 896.5788574, -551.0593262, 915.8735352, -1455.0510254, 1447.6381836
3: -928.7001953, 792.0423584, -949.8130493, 811.6398315, -1740.3400879, 1741.8554688
4: -677.5757446, 910.0938110, -693.1937866, 930.1290283, -1607.7047119, 1603.2873535

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_A2_B1_B2_A1

### Relational analysis result of NS_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2138660, upper bound: 1339.2145348
time: 0.91 seconds

## Relational analysis of NS_B1_A2_B1_B2_A2

### Relational analysis result of NS_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2138660, upper bound: 1339.2145348
time: 0.82 seconds

## BFS NS instance: NS_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -268.2239380, 242.1018372, -316.2953796, 283.6633301, -551.8872681, 558.3972168
1: -1025.0664062, 874.4836426, -1211.0053711, 1024.1472168, -2049.2136230, 2085.4890137
2: -544.2199097, 905.4760742, -639.8353882, 1062.9650879, -1607.1850586, 1545.3112793
3: -937.2784424, 799.6390381, -1104.7574463, 938.4837646, -1875.7622070, 1904.3964844
4: -683.7339478, 919.0328369, -804.7814941, 1076.7966309, -1760.5305176, 1723.8143311

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A2_B2_A1_A1

### Relational analysis result of NS_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2127811, upper bound: 1339.2144132
time: 0.91 seconds

## Relational analysis of NS_B1_A2_B2_A1_A2

### Relational analysis result of NS_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2137742, upper bound: 1339.2145348
time: 1.09 seconds

## BFS NS instance: NS_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -334.7499084, 300.0417786, -316.2953796, 283.6633301, -618.4131470, 616.3370972
1: -1282.1984863, 1083.5136719, -1211.0053711, 1024.1472168, -2306.3457031, 2294.5187988
2: -677.9343262, 1124.0098877, -639.8353882, 1062.9650879, -1740.8994141, 1763.8452148
3: -1169.1204834, 992.9312134, -1104.7574463, 938.4837646, -2107.6042480, 2097.6887207
4: -851.5982056, 1138.9154053, -804.7814941, 1076.7966309, -1928.3946533, 1943.6967773

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A2_B2_A2_B1

### Relational analysis result of NS_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2137733, upper bound: 1339.2133340
time: 1.05 seconds

## Relational analysis of NS_B1_A2_B2_A2_B2

### Relational analysis result of NS_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2137742, upper bound: 1339.2145348
time: 0.85 seconds

## BFS NS instance: NS_B2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -240.2304993, 216.8313446, -265.3952332, 239.5085449, -479.7390442, 482.2265625
1: -917.9259644, 782.8681030, -1014.6293335, 864.8557739, -1782.7817383, 1797.4974365
2: -487.6299133, 810.6806030, -539.0120239, 895.3712769, -1383.0010986, 1349.6925049
3: -839.0195312, 715.6842041, -927.1965942, 790.8286743, -1629.8480225, 1642.8808594
4: -612.4997559, 822.3607178, -676.4360962, 908.6753540, -1521.1749268, 1498.7968750

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A1_A1_A1_B1

### Relational analysis result of NS_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2132609, upper bound: 1339.2128453
time: 1.03 seconds

## Relational analysis of NS_B2_A1_A1_A1_B2

### Relational analysis result of NS_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2132609, upper bound: 1339.2138730
time: 1.11 seconds

## BFS NS instance: NS_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -271.8391724, 245.5500793, -265.7434998, 239.8792419, -511.7183838, 511.2935791
1: -1040.1838379, 887.8743286, -1015.7272949, 866.4550171, -1906.6387939, 1903.6015625
2: -551.0593262, 915.8735352, -539.1776733, 896.5788574, -1447.6381836, 1455.0510254
3: -949.8130493, 811.6398315, -928.7001953, 792.0423584, -1741.8554688, 1740.3400879
4: -693.1937866, 930.1290283, -677.5757446, 910.0938110, -1603.2873535, 1607.7047119

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_A1_A2_B1

### Relational analysis result of NS_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2132579, upper bound: 1339.2138805
time: 0.94 seconds

## Relational analysis of NS_B2_A1_A1_A2_B2

### Relational analysis result of NS_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2145348, upper bound: 1339.2138820
time: 0.99 seconds

## BFS NS instance: NS_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -316.2953796, 283.6633301, -268.2239380, 242.1018372, -558.3972168, 551.8872681
1: -1211.0053711, 1024.1472168, -1025.0664062, 874.4836426, -2085.4890137, 2049.2136230
2: -639.8353882, 1062.9650879, -544.2199097, 905.4760742, -1545.3114014, 1607.1850586
3: -1104.7574463, 938.4837646, -937.2784424, 799.6390381, -1904.3964844, 1875.7622070
4: -804.7814941, 1076.7966309, -683.7339478, 919.0328369, -1723.8143311, 1760.5305176

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A1_A2_B1_B1

### Relational analysis result of NS_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2144132, upper bound: 1339.2127811
time: 1.16 seconds

## Relational analysis of NS_B2_A1_A2_B1_B2

### Relational analysis result of NS_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2144132, upper bound: 1339.2137742
time: 1.06 seconds

## BFS NS instance: NS_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -316.2953796, 283.6633301, -334.7499084, 300.0417786, -616.3371582, 618.4132080
1: -1211.0053711, 1024.1472168, -1282.1984863, 1083.5136719, -2294.5190430, 2306.3457031
2: -639.8353882, 1062.9650879, -677.9343262, 1124.0098877, -1763.8452148, 1740.8994141
3: -1104.7574463, 938.4837646, -1169.1204834, 992.9312134, -2097.6887207, 2107.6042480
4: -804.7814941, 1076.7966309, -851.5982056, 1138.9154053, -1943.6967773, 1928.3947754

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_A1_A2_B2_A1

### Relational analysis result of NS_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2133338, upper bound: 1339.2137733
time: 0.87 seconds

## Relational analysis of NS_B2_A1_A2_B2_A2

### Relational analysis result of NS_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2145348, upper bound: 1339.2137742
time: 1.06 seconds

## BFS NS instance: NS_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -268.3889771, 242.2476959, -268.3889771, 242.2476959, -510.6366577, 510.6366577
1: -1025.7521973, 874.7044067, -1025.7521973, 874.7044067, -1900.4565430, 1900.4565430
2: -544.5684204, 906.4784546, -544.5684204, 906.4784546, -1451.0466309, 1451.0466309
3: -937.8247070, 800.2047729, -937.8247070, 800.2047729, -1738.0295410, 1738.0295410
4: -683.9780273, 919.9148560, -683.9780273, 919.9148560, -1603.8928223, 1603.8928223

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_A1_B1_B1

### Relational analysis result of NS_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2113009, upper bound: 1339.2075986
time: 1.03 seconds

## Relational analysis of NS_B2_A2_A1_B1_B2

### Relational analysis result of NS_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2130148, upper bound: 1339.2098588
time: 1.00 seconds

## BFS NS instance: NS_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -268.3889771, 242.2476959, -273.1560669, 246.9745636, -515.3635254, 515.4037476
1: -1025.7521973, 874.7044067, -1044.1036377, 891.4555054, -1917.2076416, 1918.8079834
2: -544.5684204, 906.4784546, -555.3973389, 923.8026733, -1468.3707275, 1461.8756104
3: -937.8247070, 800.2047729, -954.4480591, 815.1679077, -1752.9926758, 1754.6528320
4: -683.9780273, 919.9148560, -696.1833496, 937.7105713, -1621.6885986, 1616.0981445

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A1_B2_A1

### Relational analysis result of NS_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2137033, upper bound: 1339.2102699
time: 1.55 seconds

## Relational analysis of NS_B2_A2_A1_B2_A2

### Relational analysis result of NS_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2137033, upper bound: 1339.2101206
time: 0.90 seconds

## BFS NS instance: NS_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -273.1560669, 246.9745636, -268.3889771, 242.2476959, -515.4037476, 515.3635254
1: -1044.1036377, 891.4555054, -1025.7521973, 874.7044067, -1918.8081055, 1917.2077637
2: -555.3973389, 923.8026733, -544.5684204, 906.4784546, -1461.8756104, 1468.3707275
3: -954.4480591, 815.1679077, -937.8247070, 800.2047729, -1754.6528320, 1752.9926758
4: -696.1833496, 937.7105713, -683.9780273, 919.9148560, -1616.0981445, 1621.6885986

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A2_B1_B1

### Relational analysis result of NS_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2101759, upper bound: 1339.2100441
time: 0.94 seconds

## Relational analysis of NS_B2_A2_A2_B1_B2

### Relational analysis result of NS_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2099526, upper bound: 1339.2099526
time: 1.06 seconds

## BFS NS instance: NS_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -273.1560669, 246.9745636, -273.1560669, 246.9745636, -520.1306152, 520.1306152
1: -1044.1036377, 891.4555054, -1044.1036377, 891.4555054, -1935.5590820, 1935.5590820
2: -555.3973389, 923.8026733, -555.3973389, 923.8026733, -1479.1995850, 1479.1995850
3: -954.4480591, 815.1679077, -954.4480591, 815.1679077, -1769.6159668, 1769.6159668
4: -696.1833496, 937.7105713, -696.1833496, 937.7105713, -1633.8939209, 1633.8939209

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_A2_B2_A1

### Relational analysis result of NS_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2075410, upper bound: 1339.2091784
time: 0.89 seconds

## Relational analysis of NS_B2_A2_A2_B2_A2

### Relational analysis result of NS_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2096605, upper bound: 1339.2096605
time: 1.15 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.36 seconds
NS_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -1339.2102508, upper bound: 1339.2123506
NS_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -1339.2123150, upper bound: 1339.2138753
NS_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -1339.2129796, upper bound: 1339.2143105
NS_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -1339.2127438, upper bound: 1339.2143105
NS_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -1339.2125363, upper bound: 1339.2127501
NS_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -1339.2124666, upper bound: 1339.2124666
NS_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -1339.2116455, upper bound: 1339.2100258
NS_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -1339.2119652, upper bound: 1339.2119652
NS_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -1339.2128453, upper bound: 1339.2132609
NS_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -1339.2128453, upper bound: 1339.2132609
NS_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -1339.2138660, upper bound: 1339.2145348
NS_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -1339.2138660, upper bound: 1339.2145348
NS_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -1339.2127811, upper bound: 1339.2144132
NS_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -1339.2137742, upper bound: 1339.2145348
NS_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -1339.2137733, upper bound: 1339.2133340
NS_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -1339.2137742, upper bound: 1339.2145348
NS_B2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -1339.2132609, upper bound: 1339.2128453
NS_B2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -1339.2132609, upper bound: 1339.2138730
NS_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -1339.2132579, upper bound: 1339.2138805
NS_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -1339.2145348, upper bound: 1339.2138820
NS_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -1339.2144132, upper bound: 1339.2127811
NS_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -1339.2144132, upper bound: 1339.2137742
NS_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -1339.2133338, upper bound: 1339.2137733
NS_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -1339.2145348, upper bound: 1339.2137742
NS_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -1339.2113009, upper bound: 1339.2075986
NS_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -1339.2130148, upper bound: 1339.2098588
NS_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -1339.2137033, upper bound: 1339.2102699
NS_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -1339.2137033, upper bound: 1339.2101206
NS_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -1339.2101759, upper bound: 1339.2100441
NS_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -1339.2099526, upper bound: 1339.2099526
NS_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -1339.2075410, upper bound: 1339.2091784
NS_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -1339.2096605, upper bound: 1339.2096605

## BFS NS instance: NS_B1_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -240.8968506, 217.4349823, -244.2470093, 220.5315552, -461.4283447, 461.6820068
1: -920.6030884, 784.6614990, -933.1945190, 796.0230713, -1716.6262207, 1717.8559570
2: -488.9831848, 813.2754517, -495.6360779, 824.9326172, -1313.9155273, 1308.9114990
3: -841.3624268, 717.6347046, -853.1392212, 728.0506592, -1569.4130859, 1570.7739258
4: -614.0354004, 824.8830566, -622.5960693, 836.8617554, -1450.8968506, 1447.4791260

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_B1_A1_A1_B1

### Relational analysis result of NS_B1_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2133809, upper bound: 1339.2133809
time: 1.10 seconds

## Relational analysis of NS_B1_A1_B1_A1_A1_B2

### Relational analysis result of NS_B1_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2133809, upper bound: 1339.2145967
time: 0.88 seconds

## BFS NS instance: NS_B1_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -272.8744812, 246.4465179, -244.7990570, 221.1696320, -494.0440674, 491.2455750
1: -1044.2795410, 890.9480591, -935.0347900, 798.3237305, -1842.6032715, 1825.9825439
2: -553.1249390, 919.6119385, -496.8507690, 827.1917725, -1380.3166504, 1416.4626465
3: -953.4639893, 814.7373657, -855.4066162, 729.9403687, -1683.4042969, 1670.1440430
4: -695.6867065, 933.8435669, -624.2350464, 839.4254150, -1535.1120605, 1558.0786133

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_B1_A1_A2_B1

### Relational analysis result of NS_B1_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2145967, upper bound: 1339.2133809
time: 0.98 seconds

## Relational analysis of NS_B1_A1_B1_A1_A2_B2

### Relational analysis result of NS_B1_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2145967, upper bound: 1339.2146504
time: 1.05 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -257.0854797, 232.3377228, -246.1331482, 222.3521576, -479.4376221, 478.4708557
1: -982.2383423, 838.5811768, -939.9316406, 802.5836792, -1784.8220215, 1778.5126953
2: -522.2249146, 869.7349854, -499.5365295, 832.0221558, -1354.2470703, 1369.2714844
3: -898.1163940, 767.0713501, -859.9787598, 734.0314331, -1632.1478271, 1627.0499268
4: -655.2624512, 882.6337891, -627.5145874, 844.2769775, -1499.5394287, 1510.1481934

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2127438, upper bound: 1339.2143105
time: 1.03 seconds

## Relational analysis of NS_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2127438, upper bound: 1339.2143105
time: 1.23 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -257.1403198, 232.3919678, -312.8050232, 280.4860535, -537.6263428, 545.1969604
1: -982.4518433, 838.7551270, -1197.6622314, 1012.4920044, -1994.9437256, 2036.4172363
2: -522.3428345, 869.9675293, -632.7992554, 1051.5067139, -1573.8496094, 1502.7668457
3: -898.2748413, 767.2407837, -1092.6574707, 928.0602417, -1826.3350830, 1859.8981934
4: -655.3701782, 882.8460693, -795.8292847, 1065.0858154, -1720.4560547, 1678.6752930

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2127438, upper bound: 1339.2143105
time: 1.19 seconds

## Relational analysis of NS_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2127438, upper bound: 1339.2143105
time: 0.89 seconds

## BFS NS instance: NS_B1_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -246.1331482, 222.3521576, -257.0854797, 232.3377228, -478.4708557, 479.4376221
1: -939.9316406, 802.5836792, -982.2383423, 838.5811768, -1778.5126953, 1784.8220215
2: -499.5365295, 832.0221558, -522.2249146, 869.7349854, -1369.2714844, 1354.2470703
3: -859.9787598, 734.0314331, -898.1163940, 767.0713501, -1627.0499268, 1632.1478271
4: -627.5145874, 844.2769775, -655.2624512, 882.6337891, -1510.1481934, 1499.5394287

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_B2_A1_A1_B1

### Relational analysis result of NS_B1_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2143105, upper bound: 1339.2127438
time: 0.97 seconds

## Relational analysis of NS_B1_A1_B2_A1_A1_B2

### Relational analysis result of NS_B1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2143105, upper bound: 1339.2127438
time: 0.91 seconds

## BFS NS instance: NS_B1_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -312.8050232, 280.4860535, -257.1403198, 232.3919678, -545.1969604, 537.6263428
1: -1197.6622314, 1012.4920044, -982.4518433, 838.7551270, -2036.4172363, 1994.9437256
2: -632.7992554, 1051.5067139, -522.3428345, 869.9675293, -1502.7668457, 1573.8496094
3: -1092.6574707, 928.0602417, -898.2748413, 767.2407837, -1859.8981934, 1826.3350830
4: -795.8292847, 1065.0858154, -655.3701782, 882.8460693, -1678.6752930, 1720.4560547

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_B2_A1_A2_B1

### Relational analysis result of NS_B1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2143105, upper bound: 1339.2127438
time: 0.93 seconds

## Relational analysis of NS_B1_A1_B2_A1_A2_B2

### Relational analysis result of NS_B1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2143105, upper bound: 1339.2127438
time: 0.93 seconds

## BFS NS instance: NS_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -252.2573853, 227.9869080, -248.9954987, 224.9890442, -477.2464294, 476.9824219
1: -964.0801392, 822.5290527, -951.7622070, 811.4291992, -1775.5092773, 1774.2910156
2: -513.0211792, 853.1254883, -506.6756592, 842.1623535, -1355.1834717, 1359.8009033
3: -881.0199585, 752.3135376, -869.5330811, 742.1264648, -1623.1464844, 1621.8465576
4: -642.9034424, 865.6689453, -634.5575562, 854.2334595, -1497.1369629, 1500.2264404

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2099703, upper bound: 1339.2099703
time: 0.96 seconds

## Relational analysis of NS_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2099703, upper bound: 1339.2099836
time: 0.89 seconds

## BFS NS instance: NS_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -252.0183716, 227.6892090, -279.8794250, 252.9188843, -504.9372559, 507.5686035
1: -962.9070435, 822.0161133, -1071.2863770, 914.2707520, -1877.1777344, 1893.3024902
2: -511.6142883, 851.9074707, -567.6414185, 943.9479980, -1455.5621338, 1419.5485840
3: -880.5345459, 751.6891479, -977.8368530, 836.1256714, -1716.6599121, 1729.5260010
4: -642.5473633, 864.6139526, -713.3497925, 958.8959961, -1601.4433594, 1577.9637451

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2099836, upper bound: 1339.2116455
time: 0.92 seconds

## Relational analysis of NS_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2099836, upper bound: 1339.2119652
time: 0.96 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -262.3722229, 236.7246704, -240.2304993, 216.8313446, -479.2035522, 476.9551697
1: -1003.1879272, 854.4705200, -917.9259644, 782.8681030, -1786.0560303, 1772.3964844
2: -533.1533813, 884.9224243, -487.6299133, 810.6806030, -1343.8336182, 1372.5523682
3: -916.5409546, 781.3559570, -839.0195312, 715.6842041, -1632.2250977, 1620.3753662
4: -668.7097168, 897.9597778, -612.4997559, 822.3607178, -1491.0704346, 1510.4594727

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B1_A2_B1_B1_A1_B1

### Relational analysis result of NS_B1_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2075901, upper bound: 1339.2110928
time: 0.83 seconds

## Relational analysis of NS_B1_A2_B1_B1_A1_B2

### Relational analysis result of NS_B1_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2075372, upper bound: 1339.2096301
time: 0.84 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -292.6012268, 264.1650391, -240.2304993, 216.8313446, -509.4325256, 504.3955383
1: -1120.2954102, 955.0587158, -917.9259644, 782.8681030, -1903.1635742, 1872.9846191
2: -592.9663086, 984.5103760, -487.6299133, 810.6806030, -1403.6466064, 1472.1402588
3: -1022.6113281, 873.0273438, -839.0195312, 715.6842041, -1738.2955322, 1712.0466309
4: -746.0636597, 1000.0946655, -612.4997559, 822.3607178, -1568.4243164, 1612.5944824

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_B1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_B1_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2124705, upper bound: 1339.2133108
time: 3.68 seconds

## Relational analysis of NS_B1_A2_B1_B1_A2_B2

### Relational analysis result of NS_B1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2128120, upper bound: 1339.2133370
time: 1.11 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -262.7026672, 237.1206665, -271.8391724, 245.5500793, -508.2527466, 508.9597778
1: -1004.0178223, 856.5154419, -1040.1838379, 887.8743286, -1891.8920898, 1896.6990967
2: -533.0585327, 886.3162231, -551.0593262, 915.8735352, -1448.9317627, 1437.3754883
3: -918.1187744, 782.9194946, -949.8130493, 811.6398315, -1729.7585449, 1732.7321777
4: -669.8895264, 899.7063599, -693.1937866, 930.1290283, -1600.0185547, 1592.9001465

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A2_B1_B2_A1_A1

### Relational analysis result of NS_B1_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2128453, upper bound: 1339.2144132
time: 1.17 seconds

## Relational analysis of NS_B1_A2_B1_B2_A1_A2

### Relational analysis result of NS_B1_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2128453, upper bound: 1339.2142815
time: 0.96 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -329.4176331, 295.2451477, -271.8391724, 245.5500793, -574.9677124, 567.0843506
1: -1261.8427734, 1066.0521240, -1040.1838379, 887.8743286, -2149.7167969, 2106.2358398
2: -666.7619019, 1105.5648193, -551.0593262, 915.8735352, -1582.6351318, 1656.6241455
3: -1150.6239014, 976.7224731, -949.8130493, 811.6398315, -1962.2636719, 1926.5355225
4: -838.2064209, 1120.1811523, -693.1937866, 930.1290283, -1768.3354492, 1813.3748779

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A2_B1_B2_A2_A1

### Relational analysis result of NS_B1_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2128453, upper bound: 1339.2144132
time: 0.93 seconds

## Relational analysis of NS_B1_A2_B1_B2_A2_A2

### Relational analysis result of NS_B1_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2128453, upper bound: 1339.2142815
time: 0.99 seconds

## BFS NS instance: NS_B1_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -259.3377991, 233.9734497, -310.5607910, 278.3730774, -537.7108765, 544.5342407
1: -991.4920044, 844.5764160, -1189.2741699, 1005.0030518, -1996.4948730, 2033.8504639
2: -527.0584717, 874.6990967, -628.3911743, 1043.0084229, -1570.0667725, 1503.0903320
3: -905.9906616, 772.2731934, -1084.5045166, 920.9031982, -1826.8936768, 1856.7777100
4: -661.0531006, 887.6106567, -790.1387329, 1056.3688965, -1717.4219971, 1677.7492676

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A2_B2_A1_A1_B1

### Relational analysis result of NS_B1_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2126911, upper bound: 1339.2133521
time: 1.15 seconds

## Relational analysis of NS_B1_A2_B2_A1_A1_B2

### Relational analysis result of NS_B1_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2126911, upper bound: 1339.2144893
time: 1.11 seconds

## BFS NS instance: NS_B1_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -289.5389709, 261.3972778, -310.9486694, 278.8442688, -568.3832397, 572.3459473
1: -1108.5106201, 945.0742798, -1190.6361084, 1006.6168823, -2115.1274414, 2135.7104492
2: -586.8018188, 974.2145996, -629.0652466, 1044.4067383, -1631.2083740, 1603.2794189
3: -1011.9727783, 863.8530273, -1086.2800293, 922.2344971, -1934.2071533, 1950.1330566
4: -738.3388062, 989.6480103, -791.3406372, 1058.0332031, -1796.3719482, 1780.9886475

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A2_B2_A1_A2_B1

### Relational analysis result of NS_B1_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2137733, upper bound: 1339.2133521
time: 1.00 seconds

## Relational analysis of NS_B1_A2_B2_A1_A2_B2

### Relational analysis result of NS_B1_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2137733, upper bound: 1339.2146504
time: 1.15 seconds

## BFS NS instance: NS_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -329.3286133, 295.0782166, -307.4197388, 275.4576416, -604.7862549, 602.4979248
1: -1261.6467285, 1065.3712158, -1177.4262695, 994.2976074, -2255.9443359, 2242.7973633
2: -667.4924927, 1105.3580322, -622.2865601, 1031.9949951, -1699.4875488, 1727.6442871
3: -1149.9514160, 976.3134155, -1073.4550781, 911.1184082, -2061.0698242, 2049.7683105
4: -837.7307129, 1119.7712402, -782.1300659, 1045.1027832, -1882.8333740, 1901.9013672

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2126911, upper bound: 1339.2132345
time: 1.05 seconds

## Relational analysis of NS_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2126911, upper bound: 1339.2132345
time: 1.08 seconds

## BFS NS instance: NS_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -329.4176331, 295.2451477, -339.2242126, 304.3148193, -633.7324219, 634.4693604
1: -1261.8427734, 1066.0521240, -1300.6374512, 1099.3801270, -2361.2229004, 2366.6894531
2: -666.7619019, 1105.5648193, -685.9866943, 1137.1968994, -1803.9584961, 1791.5515137
3: -1150.6239014, 976.7224731, -1185.0489502, 1007.4972534, -2158.1210938, 2161.7712402
4: -838.2064209, 1120.1811523, -863.5870361, 1153.0744629, -1991.2807617, 1983.7681885

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2126911, upper bound: 1339.2144132
time: 0.92 seconds

## Relational analysis of NS_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2126911, upper bound: 1339.2145348
time: 0.92 seconds

## BFS NS instance: NS_B2_A1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -240.2304993, 216.8313446, -262.3722229, 236.7246704, -476.9551697, 479.2035522
1: -917.9259644, 782.8681030, -1003.1879272, 854.4705200, -1772.3964844, 1786.0560303
2: -487.6299133, 810.6806030, -533.1533813, 884.9224243, -1372.5523682, 1343.8337402
3: -839.0195312, 715.6842041, -916.5409546, 781.3559570, -1620.3753662, 1632.2250977
4: -612.4997559, 822.3607178, -668.7097168, 897.9597778, -1510.4594727, 1491.0703125

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B2_A1_A1_A1_B1_A1

### Relational analysis result of NS_B2_A1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2110928, upper bound: 1339.2075901
time: 0.96 seconds

## Relational analysis of NS_B2_A1_A1_A1_B1_A2

### Relational analysis result of NS_B2_A1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2096301, upper bound: 1339.2075372
time: 0.90 seconds

## BFS NS instance: NS_B2_A1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -240.2304993, 216.8313446, -292.6012268, 264.1650391, -504.3955383, 509.4325256
1: -917.9259644, 782.8681030, -1120.2954102, 955.0587158, -1872.9846191, 1903.1635742
2: -487.6299133, 810.6806030, -592.9663086, 984.5103760, -1472.1402588, 1403.6466064
3: -839.0195312, 715.6842041, -1022.6113281, 873.0273438, -1712.0466309, 1738.2955322
4: -612.4997559, 822.3607178, -746.0636597, 1000.0946655, -1612.5944824, 1568.4243164

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_B2_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_A1_A1_B2_A1

### Relational analysis result of NS_B2_A1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2133108, upper bound: 1339.2135892
time: 0.85 seconds

## Relational analysis of NS_B2_A1_A1_A1_B2_A2

### Relational analysis result of NS_B2_A1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2133370, upper bound: 1339.2138730
time: 1.23 seconds

## BFS NS instance: NS_B2_A1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -271.8391724, 245.5500793, -262.7026672, 237.1206665, -508.9598083, 508.2527466
1: -1040.1838379, 887.8743286, -1004.0178223, 856.5154419, -1896.6990967, 1891.8920898
2: -551.0593262, 915.8735352, -533.0585327, 886.3162231, -1437.3754883, 1448.9317627
3: -949.8130493, 811.6398315, -918.1187744, 782.9194946, -1732.7321777, 1729.7585449
4: -693.1937866, 930.1290283, -669.8895264, 899.7063599, -1592.9001465, 1600.0185547

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A1_A1_A2_B1_B1

### Relational analysis result of NS_B2_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2144132, upper bound: 1339.2128453
time: 1.03 seconds

## Relational analysis of NS_B2_A1_A1_A2_B1_B2

### Relational analysis result of NS_B2_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2144132, upper bound: 1339.2136871
time: 0.96 seconds

## BFS NS instance: NS_B2_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -271.8391724, 245.5500793, -329.4176331, 295.2451477, -567.0843506, 574.9677124
1: -1040.1838379, 887.8743286, -1261.8427734, 1066.0521240, -2106.2358398, 2149.7167969
2: -551.0593262, 915.8735352, -666.7619019, 1105.5648193, -1656.6241455, 1582.6350098
3: -949.8130493, 811.6398315, -1150.6239014, 976.7224731, -1926.5355225, 1962.2636719
4: -693.1937866, 930.1290283, -838.2064209, 1120.1811523, -1813.3747559, 1768.3354492

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A1_A1_A2_B2_B1

### Relational analysis result of NS_B2_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2144132, upper bound: 1339.2128453
time: 1.21 seconds

## Relational analysis of NS_B2_A1_A1_A2_B2_B2

### Relational analysis result of NS_B2_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2144132, upper bound: 1339.2136926
time: 1.00 seconds

## BFS NS instance: NS_B2_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -310.5607910, 278.3730774, -259.3377991, 233.9734497, -544.5342407, 537.7108765
1: -1189.2741699, 1005.0030518, -991.4920044, 844.5764160, -2033.8504639, 1996.4948730
2: -628.3911743, 1043.0084229, -527.0584717, 874.6990967, -1503.0903320, 1570.0666504
3: -1084.5045166, 920.9031982, -905.9906616, 772.2731934, -1856.7777100, 1826.8936768
4: -790.1387329, 1056.3688965, -661.0531006, 887.6106567, -1677.7491455, 1717.4219971

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_A1_A2_B1_B1_A1

### Relational analysis result of NS_B2_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2133521, upper bound: 1339.2126911
time: 0.83 seconds

## Relational analysis of NS_B2_A1_A2_B1_B1_A2

### Relational analysis result of NS_B2_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2133521, upper bound: 1339.2126911
time: 1.25 seconds

## BFS NS instance: NS_B2_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -310.9486694, 278.8442688, -289.5389709, 261.3972778, -572.3459473, 568.3832397
1: -1190.6361084, 1006.6168823, -1108.5106201, 945.0742798, -2135.7102051, 2115.1274414
2: -629.0652466, 1044.4067383, -586.8018188, 974.2145996, -1603.2795410, 1631.2083740
3: -1086.2800293, 922.2344971, -1011.9727783, 863.8530273, -1950.1330566, 1934.2071533
4: -791.3406372, 1058.0332031, -738.3388062, 989.6480103, -1780.9886475, 1796.3720703

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_A1_A2_B1_B2_A1

### Relational analysis result of NS_B2_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2133521, upper bound: 1339.2137733
time: 1.04 seconds

## Relational analysis of NS_B2_A1_A2_B1_B2_A2

### Relational analysis result of NS_B2_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2133521, upper bound: 1339.2137742
time: 1.10 seconds

## BFS NS instance: NS_B2_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -307.4197388, 275.4576416, -329.3286133, 295.0782166, -602.4979248, 604.7862549
1: -1177.4262695, 994.2976074, -1261.6467285, 1065.3712158, -2242.7973633, 2255.9443359
2: -622.2865601, 1031.9949951, -667.4924927, 1105.3580322, -1727.6441650, 1699.4875488
3: -1073.4550781, 911.1184082, -1149.9514160, 976.3134155, -2049.7685547, 2061.0698242
4: -782.1300659, 1045.1027832, -837.7307129, 1119.7712402, -1901.9013672, 1882.8333740

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A1_A2_B2_A1_B1

### Relational analysis result of NS_B2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2132271, upper bound: 1339.2126911
time: 1.07 seconds

## Relational analysis of NS_B2_A1_A2_B2_A1_B2

### Relational analysis result of NS_B2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2132271, upper bound: 1339.2137733
time: 1.02 seconds

## BFS NS instance: NS_B2_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -339.2242126, 304.3148193, -329.4176331, 295.2451477, -634.4693604, 633.7324219
1: -1300.6374512, 1099.3801270, -1261.8427734, 1066.0521240, -2366.6894531, 2361.2229004
2: -685.9866943, 1137.1968994, -666.7619019, 1105.5648193, -1791.5515137, 1803.9584961
3: -1185.0489502, 1007.4972534, -1150.6239014, 976.7224731, -2161.7712402, 2158.1210938
4: -863.5870361, 1153.0744629, -838.2064209, 1120.1811523, -1983.7681885, 1991.2808838

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A1_A2_B2_A2_B1

### Relational analysis result of NS_B2_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2144132, upper bound: 1339.2126911
time: 1.29 seconds

## Relational analysis of NS_B2_A1_A2_B2_A2_B2

### Relational analysis result of NS_B2_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2144132, upper bound: 1339.2137742
time: 0.91 seconds

## BFS NS instance: NS_B2_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -262.5783997, 236.9383545, -259.6002808, 234.1820068, -496.7603149, 496.5385437
1: -1003.8460693, 855.3237305, -992.5734863, 845.0814209, -1848.9273682, 1847.8970947
2: -533.3724976, 886.3398438, -527.6042480, 876.0423584, -1409.4147949, 1413.9439697
3: -917.3871460, 782.4293823, -906.8853760, 773.0839233, -1690.4710693, 1689.3146973
4: -669.1592407, 899.3970947, -661.5457764, 888.8338623, -1557.9929199, 1560.9428711

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_A1_B1_B1_A1

### Relational analysis result of NS_B2_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2128453, upper bound: 1339.2128453
time: 1.15 seconds

## Relational analysis of NS_B2_A2_A1_B1_B1_A2

### Relational analysis result of NS_B2_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2128453, upper bound: 1339.2128453
time: 0.95 seconds

## BFS NS instance: NS_B2_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -262.8648376, 237.2611847, -289.9319153, 261.7218933, -524.5866699, 527.1930542
1: -1004.7023926, 856.7432251, -1110.0667725, 946.1136475, -1950.8160400, 1966.8098145
2: -533.4082031, 887.3182373, -587.6096802, 975.8930054, -1509.3011475, 1474.9279785
3: -918.6716919, 783.4802856, -1013.3492432, 865.1301270, -1783.8017578, 1796.8294678
4: -670.1300049, 900.5930786, -739.1745605, 991.2822876, -1661.4123535, 1639.7672119

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_A1_B1_B2_A1

### Relational analysis result of NS_B2_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2128453, upper bound: 1339.2138730
time: 1.16 seconds

## Relational analysis of NS_B2_A2_A1_B1_B2_A2

### Relational analysis result of NS_B2_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2128453, upper bound: 1339.2138820
time: 0.87 seconds

## BFS NS instance: NS_B2_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -264.2924500, 238.5483704, -272.0606689, 245.9839172, -510.2763367, 510.6090393
1: -1009.9717407, 861.3646240, -1039.8781738, 887.8885498, -1897.8602295, 1901.2427979
2: -536.3375244, 892.6682129, -553.1947632, 920.0974731, -1456.4350586, 1445.8629150
3: -923.5935059, 787.9431152, -950.6499023, 811.8846436, -1735.4781494, 1738.5928955
4: -673.6499634, 905.9495850, -693.4251709, 933.9655151, -1607.6154785, 1599.3747559

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A1_B2_A1_B1

### Relational analysis result of NS_B2_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2137033, upper bound: 1339.2101206
time: 1.18 seconds

## Relational analysis of NS_B2_A2_A1_B2_A1_B2

### Relational analysis result of NS_B2_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2137033, upper bound: 1339.2101206
time: 0.87 seconds

## BFS NS instance: NS_B2_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -331.0111084, 296.6192322, -272.2126770, 246.1097260, -577.1207275, 568.8319092
1: -1267.8787842, 1071.0562744, -1040.4826660, 888.3626709, -2156.2414551, 2111.5390625
2: -670.4309082, 1111.8437500, -553.4670410, 920.5850830, -1591.0157471, 1665.3107910
3: -1156.1138916, 981.8455200, -951.1641235, 812.3130493, -1968.4265137, 1933.0095215
4: -841.9860840, 1126.4869385, -693.7783203, 934.4380493, -1776.4240723, 1820.2652588

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A1_B2_A2_B1

### Relational analysis result of NS_B2_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2137033, upper bound: 1339.2101206
time: 1.08 seconds

## Relational analysis of NS_B2_A2_A1_B2_A2_B2

### Relational analysis result of NS_B2_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2137033, upper bound: 1339.2101206
time: 1.02 seconds

## BFS NS instance: NS_B2_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -272.0606689, 245.9839172, -264.2924500, 238.5483704, -510.6090393, 510.2763367
1: -1039.8781738, 887.8885498, -1009.9717407, 861.3646240, -1901.2427979, 1897.8602295
2: -553.1947632, 920.0974731, -536.3375244, 892.6682129, -1445.8629150, 1456.4350586
3: -950.6499023, 811.8846436, -923.5935059, 787.9431152, -1738.5926514, 1735.4781494
4: -693.4251709, 933.9655151, -673.6499634, 905.9495850, -1599.3747559, 1607.6154785

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A2_B1_B1_A1

### Relational analysis result of NS_B2_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2101070, upper bound: 1339.2137033
time: 1.16 seconds

## Relational analysis of NS_B2_A2_A2_B1_B1_A2

### Relational analysis result of NS_B2_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2101070, upper bound: 1339.2137033
time: 0.84 seconds

## BFS NS instance: NS_B2_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -272.2126770, 246.1097260, -331.0111084, 296.6192322, -568.8319092, 577.1208496
1: -1040.4826660, 888.3626709, -1267.8787842, 1071.0562744, -2111.5390625, 2156.2412109
2: -553.4670410, 920.5850830, -670.4309082, 1111.8437500, -1665.3107910, 1591.0157471
3: -951.1641235, 812.3130493, -1156.1138916, 981.8455200, -1933.0096436, 1968.4265137
4: -693.7783203, 934.4380493, -841.9860840, 1126.4869385, -1820.2652588, 1776.4240723

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A2_B1_B2_A1

### Relational analysis result of NS_B2_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2101070, upper bound: 1339.2137033
time: 1.12 seconds

## Relational analysis of NS_B2_A2_A2_B1_B2_A2

### Relational analysis result of NS_B2_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2101070, upper bound: 1339.2137033
time: 1.04 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -264.6048584, 239.1730957, -267.4748840, 241.8156433, -506.4204712, 506.6479797
1: -1011.7265015, 862.8225098, -1022.6122437, 872.5455933, -1884.2720947, 1885.4346924
2: -538.8558960, 895.0886230, -544.4107666, 904.4436646, -1443.2994385, 1439.4993896
3: -924.2379761, 788.8666382, -934.3807373, 797.7935181, -1722.0314941, 1723.2471924
4: -674.3473511, 907.8676147, -681.6704712, 917.8596191, -1592.2069092, 1589.5377197

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_A2_B2_A1_B1

### Relational analysis result of NS_B2_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2075372, upper bound: 1339.2075372
time: 1.02 seconds

## Relational analysis of NS_B2_A2_A2_B2_A1_B2

### Relational analysis result of NS_B2_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2075372, upper bound: 1339.2091784
time: 1.20 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -293.3989258, 265.1939087, -267.0506287, 241.3401947, -534.7391357, 532.2443848
1: -1123.3011475, 958.5373535, -1020.7564087, 871.5045166, -1994.8056641, 1979.2934570
2: -595.5602417, 988.7298584, -542.2412720, 902.3704224, -1497.9305420, 1530.9711914
3: -1025.4193115, 876.1559448, -933.2692261, 796.6746216, -1822.0938721, 1809.4251709
4: -748.0745239, 1004.4883423, -680.8818970, 915.9735718, -1664.0480957, 1685.3702393

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_A2_B2_A2_B1

### Relational analysis result of NS_B2_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2091784, upper bound: 1339.2075372
time: 1.00 seconds

## Relational analysis of NS_B2_A2_A2_B2_A2_B2

### Relational analysis result of NS_B2_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2091784, upper bound: 1339.2096605
time: 1.01 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.12 seconds
NS_B1_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2133809, upper bound: 1339.2133809
NS_B1_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2133809, upper bound: 1339.2145967
NS_B1_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2145967, upper bound: 1339.2133809
NS_B1_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2145967, upper bound: 1339.2146504
NS_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2127438, upper bound: 1339.2143105
NS_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2127438, upper bound: 1339.2143105
NS_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2127438, upper bound: 1339.2143105
NS_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2127438, upper bound: 1339.2143105
NS_B1_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2143105, upper bound: 1339.2127438
NS_B1_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2143105, upper bound: 1339.2127438
NS_B1_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2143105, upper bound: 1339.2127438
NS_B1_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2143105, upper bound: 1339.2127438
NS_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2099703, upper bound: 1339.2099703
NS_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2099703, upper bound: 1339.2099836
NS_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2099836, upper bound: 1339.2116455
NS_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2099836, upper bound: 1339.2119652
NS_B1_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2075901, upper bound: 1339.2110928
NS_B1_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2075372, upper bound: 1339.2096301
NS_B1_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2124705, upper bound: 1339.2133108
NS_B1_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2128120, upper bound: 1339.2133370
NS_B1_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2128453, upper bound: 1339.2144132
NS_B1_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2128453, upper bound: 1339.2142815
NS_B1_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2128453, upper bound: 1339.2144132
NS_B1_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2128453, upper bound: 1339.2142815
NS_B1_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2126911, upper bound: 1339.2133521
NS_B1_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2126911, upper bound: 1339.2144893
NS_B1_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2137733, upper bound: 1339.2133521
NS_B1_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2137733, upper bound: 1339.2146504
NS_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2126911, upper bound: 1339.2132345
NS_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2126911, upper bound: 1339.2132345
NS_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2126911, upper bound: 1339.2144132
NS_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2126911, upper bound: 1339.2145348
NS_B2_A1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2110928, upper bound: 1339.2075901
NS_B2_A1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2096301, upper bound: 1339.2075372
NS_B2_A1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2133108, upper bound: 1339.2135892
NS_B2_A1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2133370, upper bound: 1339.2138730
NS_B2_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2144132, upper bound: 1339.2128453
NS_B2_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2144132, upper bound: 1339.2136871
NS_B2_A1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2144132, upper bound: 1339.2128453
NS_B2_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2144132, upper bound: 1339.2136926
NS_B2_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2133521, upper bound: 1339.2126911
NS_B2_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2133521, upper bound: 1339.2126911
NS_B2_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2133521, upper bound: 1339.2137733
NS_B2_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2133521, upper bound: 1339.2137742
NS_B2_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2132271, upper bound: 1339.2126911
NS_B2_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2132271, upper bound: 1339.2137733
NS_B2_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2144132, upper bound: 1339.2126911
NS_B2_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2144132, upper bound: 1339.2137742
NS_B2_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2128453, upper bound: 1339.2128453
NS_B2_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2128453, upper bound: 1339.2128453
NS_B2_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2128453, upper bound: 1339.2138730
NS_B2_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2128453, upper bound: 1339.2138820
NS_B2_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2137033, upper bound: 1339.2101206
NS_B2_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2137033, upper bound: 1339.2101206
NS_B2_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2137033, upper bound: 1339.2101206
NS_B2_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2137033, upper bound: 1339.2101206
NS_B2_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2101070, upper bound: 1339.2137033
NS_B2_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2101070, upper bound: 1339.2137033
NS_B2_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2101070, upper bound: 1339.2137033
NS_B2_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2101070, upper bound: 1339.2137033
NS_B2_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2075372, upper bound: 1339.2075372
NS_B2_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2075372, upper bound: 1339.2091784
NS_B2_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2091784, upper bound: 1339.2075372
NS_B2_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 3, lower bound: -1339.2091784, upper bound: 1339.2096605

## BFS NS instance: NS_B1_A1_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -240.8968506, 217.4349823, -240.8968506, 217.4349823, -458.3318481, 458.3318481
1: -920.6030884, 784.6614990, -920.6030884, 784.6614990, -1705.2646484, 1705.2646484
2: -488.9831848, 813.2754517, -488.9831848, 813.2754517, -1302.2586670, 1302.2586670
3: -841.3624268, 717.6347046, -841.3624268, 717.6347046, -1558.9970703, 1558.9970703
4: -614.0354004, 824.8830566, -614.0354004, 824.8830566, -1438.9184570, 1438.9184570

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_B1_A1_A1_B1_B1

### Relational analysis result of NS_B1_A1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2134687, upper bound: 1339.2132615
time: 0.98 seconds

## Relational analysis of NS_B1_A1_B1_A1_A1_B1_B2

### Relational analysis result of NS_B1_A1_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2133338, upper bound: 1339.2132317
time: 1.20 seconds

## BFS NS instance: NS_B1_A1_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -240.8968506, 217.4349823, -272.8744812, 246.4465179, -487.3433838, 490.3094482
1: -920.6030884, 784.6614990, -1044.2795410, 890.9480591, -1811.5511475, 1828.9410400
2: -488.9831848, 813.2754517, -553.1249390, 919.6119385, -1408.5950928, 1366.4003906
3: -841.3624268, 717.6347046, -953.4639893, 814.7373657, -1656.0998535, 1671.0986328
4: -614.0354004, 824.8830566, -695.6867065, 933.8435669, -1547.8789062, 1520.5698242

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_B1_A1_A1_B2_A1

### Relational analysis result of NS_B1_A1_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2133462, upper bound: 1339.2145964
time: 1.09 seconds

## Relational analysis of NS_B1_A1_B1_A1_A1_B2_A2

### Relational analysis result of NS_B1_A1_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2133338, upper bound: 1339.2144956
time: 0.94 seconds

## BFS NS instance: NS_B1_A1_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -272.8744812, 246.4465179, -240.8968506, 217.4349823, -490.3094482, 487.3433838
1: -1044.2795410, 890.9480591, -920.6030884, 784.6614990, -1828.9410400, 1811.5511475
2: -553.1249390, 919.6119385, -488.9831848, 813.2754517, -1366.4003906, 1408.5950928
3: -953.4639893, 814.7373657, -841.3624268, 717.6347046, -1671.0986328, 1656.0998535
4: -695.6867065, 933.8435669, -614.0354004, 824.8830566, -1520.5698242, 1547.8789062

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_B1_A1_A2_B1_B1

### Relational analysis result of NS_B1_A1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2145964, upper bound: 1339.2132579
time: 0.98 seconds

## Relational analysis of NS_B1_A1_B1_A1_A2_B1_B2

### Relational analysis result of NS_B1_A1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2144956, upper bound: 1339.2132271
time: 1.11 seconds

## BFS NS instance: NS_B1_A1_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -272.8744812, 246.4465179, -272.8744812, 246.4465179, -519.3209839, 519.3209839
1: -1044.2795410, 890.9480591, -1044.2795410, 890.9480591, -1935.2275391, 1935.2275391
2: -553.1249390, 919.6119385, -553.1249390, 919.6119385, -1472.7368164, 1472.7368164
3: -953.4639893, 814.7373657, -953.4639893, 814.7373657, -1768.2014160, 1768.2014160
4: -695.6867065, 933.8435669, -695.6867065, 933.8435669, -1629.5302734, 1629.5302734

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_B1_A1_A2_B2_B1

### Relational analysis result of NS_B1_A1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2145964, upper bound: 1339.2142816
time: 1.25 seconds

## Relational analysis of NS_B1_A1_B1_A1_A2_B2_B2

### Relational analysis result of NS_B1_A1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2144956, upper bound: 1339.2142816
time: 1.07 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -253.9728699, 229.5139923, -246.1331482, 222.3521576, -476.3250122, 475.6470947
1: -970.2525024, 828.4385986, -939.9316406, 802.5836792, -1772.8360596, 1768.3702393
2: -515.9530640, 859.1966553, -499.5365295, 832.0221558, -1347.9749756, 1358.7329102
3: -887.3055420, 757.7397461, -859.9787598, 734.0314331, -1621.3369141, 1617.7182617
4: -647.4060059, 871.9716797, -627.5145874, 844.2769775, -1491.6828613, 1499.4860840

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2121613, upper bound: 1339.2115290
time: 0.99 seconds

## Relational analysis of NS_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2122811, upper bound: 1339.2134812
time: 1.02 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -320.0093689, 287.1600647, -246.1331482, 222.3521576, -542.3615112, 533.2932129
1: -1225.3875732, 1036.5230713, -939.9316406, 802.5836792, -2027.9711914, 1976.4547119
2: -648.1471558, 1076.3354492, -499.5365295, 832.0221558, -1480.1693115, 1575.8719482
3: -1117.9532471, 950.0099487, -859.9787598, 734.0314331, -1851.9846191, 1809.9885254
4: -814.0174561, 1090.8045654, -627.5145874, 844.2769775, -1658.2944336, 1718.3190918

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2121613, upper bound: 1339.2115290
time: 1.14 seconds

## Relational analysis of NS_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2122811, upper bound: 1339.2134812
time: 1.11 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -253.9728699, 229.5139923, -312.8050232, 280.4860535, -534.4589233, 542.3190308
1: -970.2525024, 828.4385986, -1197.6622314, 1012.4920044, -1982.7443848, 2026.1004639
2: -515.9530640, 859.1966553, -632.7992554, 1051.5067139, -1567.4594727, 1491.9958496
3: -887.3055420, 757.7397461, -1092.6574707, 928.0602417, -1815.3656006, 1850.3970947
4: -647.4060059, 871.9716797, -795.8292847, 1065.0858154, -1712.4914551, 1667.8010254

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_B1_A2_B2_A1_A1

### Relational analysis result of NS_B1_A1_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2100063, upper bound: 1339.2119360
time: 0.92 seconds

## Relational analysis of NS_B1_A1_B1_A2_B2_A1_A2

### Relational analysis result of NS_B1_A1_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2120553, upper bound: 1339.2134812
time: 0.92 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -320.0093689, 287.1600647, -312.8050232, 280.4860535, -600.4954224, 599.9650879
1: -1225.3875732, 1036.5230713, -1197.6622314, 1012.4920044, -2237.8796387, 2234.1853027
2: -648.1471558, 1076.3354492, -632.7992554, 1051.5067139, -1699.6538086, 1709.1347656
3: -1117.9532471, 950.0099487, -1092.6574707, 928.0602417, -2046.0133057, 2042.6673584
4: -814.0174561, 1090.8045654, -795.8292847, 1065.0858154, -1879.1030273, 1886.6337891

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2119690, upper bound: 1339.2114797
time: 0.88 seconds

## Relational analysis of NS_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2120553, upper bound: 1339.2134812
time: 1.02 seconds

## BFS NS instance: NS_B1_A1_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -246.1331482, 222.3521576, -253.9728699, 229.5139923, -475.6470947, 476.3250122
1: -939.9316406, 802.5836792, -970.2525024, 828.4385986, -1768.3702393, 1772.8359375
2: -499.5365295, 832.0221558, -515.9530640, 859.1966553, -1358.7329102, 1347.9749756
3: -859.9787598, 734.0314331, -887.3055420, 757.7397461, -1617.7182617, 1621.3369141
4: -627.5145874, 844.2769775, -647.4060059, 871.9716797, -1499.4862061, 1491.6828613

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_B2_A1_A1_B1_A1

### Relational analysis result of NS_B1_A1_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2115290, upper bound: 1339.2121613
time: 0.89 seconds

## Relational analysis of NS_B1_A1_B2_A1_A1_B1_A2

### Relational analysis result of NS_B1_A1_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2134812, upper bound: 1339.2122811
time: 1.02 seconds

## BFS NS instance: NS_B1_A1_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -246.1331482, 222.3521576, -320.0093689, 287.1600647, -533.2932129, 542.3615112
1: -939.9316406, 802.5836792, -1225.3875732, 1036.5230713, -1976.4547119, 2027.9711914
2: -499.5365295, 832.0221558, -648.1471558, 1076.3354492, -1575.8719482, 1480.1693115
3: -859.9787598, 734.0314331, -1117.9532471, 950.0099487, -1809.9884033, 1851.9846191
4: -627.5145874, 844.2769775, -814.0174561, 1090.8045654, -1718.3189697, 1658.2944336

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_B2_A1_A1_B2_A1

### Relational analysis result of NS_B1_A1_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2115290, upper bound: 1339.2121613
time: 1.04 seconds

## Relational analysis of NS_B1_A1_B2_A1_A1_B2_A2

### Relational analysis result of NS_B1_A1_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2134812, upper bound: 1339.2122811
time: 1.12 seconds

## BFS NS instance: NS_B1_A1_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -312.8050232, 280.4860535, -253.9728699, 229.5139923, -542.3190308, 534.4589233
1: -1197.6622314, 1012.4920044, -970.2525024, 828.4385986, -2026.1004639, 1982.7443848
2: -632.7992554, 1051.5067139, -515.9530640, 859.1966553, -1491.9958496, 1567.4594727
3: -1092.6574707, 928.0602417, -887.3055420, 757.7397461, -1850.3969727, 1815.3654785
4: -795.8292847, 1065.0858154, -647.4060059, 871.9716797, -1667.8010254, 1712.4914551

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_B2_A1_A2_B1_B1

### Relational analysis result of NS_B1_A1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2119360, upper bound: 1339.2100063
time: 1.13 seconds

## Relational analysis of NS_B1_A1_B2_A1_A2_B1_B2

### Relational analysis result of NS_B1_A1_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2134812, upper bound: 1339.2120553
time: 1.09 seconds

## BFS NS instance: NS_B1_A1_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -312.8050232, 280.4860535, -320.0093689, 287.1600647, -599.9650879, 600.4954224
1: -1197.6622314, 1012.4920044, -1225.3875732, 1036.5230713, -2234.1853027, 2237.8793945
2: -632.7992554, 1051.5067139, -648.1471558, 1076.3354492, -1709.1347656, 1699.6538086
3: -1092.6574707, 928.0602417, -1117.9532471, 950.0099487, -2042.6672363, 2046.0133057
4: -795.8292847, 1065.0858154, -814.0174561, 1090.8045654, -1886.6337891, 1879.1030273

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_B2_A1_A2_B2_A1

### Relational analysis result of NS_B1_A1_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2114797, upper bound: 1339.2119690
time: 1.10 seconds

## Relational analysis of NS_B1_A1_B2_A1_A2_B2_A2

### Relational analysis result of NS_B1_A1_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2134812, upper bound: 1339.2120553
time: 1.12 seconds

## BFS NS instance: NS_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -248.9954987, 224.9890442, -248.9954987, 224.9890442, -473.9845276, 473.9844971
1: -951.7622070, 811.4291992, -951.7622070, 811.4291992, -1763.1911621, 1763.1911621
2: -506.6756592, 842.1623535, -506.6756592, 842.1623535, -1348.8377686, 1348.8378906
3: -869.5330811, 742.1264648, -869.5330811, 742.1264648, -1611.6595459, 1611.6595459
4: -634.5575562, 854.2334595, -634.5575562, 854.2334595, -1488.7910156, 1488.7910156

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_B2_A2_B1_A1_A1

### Relational analysis result of NS_B1_A1_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2097285, upper bound: 1339.2099457
time: 1.12 seconds

## Relational analysis of NS_B1_A1_B2_A2_B1_A1_A2

### Relational analysis result of NS_B1_A1_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2096757, upper bound: 1339.2097077
time: 0.96 seconds

## BFS NS instance: NS_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -279.8794250, 252.9188843, -248.9954987, 224.9890442, -504.8684692, 501.9143677
1: -1071.2863770, 914.2707520, -951.7622070, 811.4291992, -1882.7155762, 1866.0329590
2: -567.6414185, 943.9479980, -506.6756592, 842.1623535, -1409.8037109, 1450.6235352
3: -977.8368530, 836.1256714, -869.5330811, 742.1264648, -1719.9633789, 1705.6585693
4: -713.3497925, 958.8959961, -634.5575562, 854.2334595, -1567.5832520, 1593.4536133

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2099148, upper bound: 1339.2097890
time: 0.95 seconds

## Relational analysis of NS_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2096757, upper bound: 1339.2097077
time: 0.89 seconds

## BFS NS instance: NS_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -248.9954987, 224.9890442, -279.8794250, 252.9188843, -501.9143677, 504.8684692
1: -951.7622070, 811.4291992, -1071.2863770, 914.2707520, -1866.0328369, 1882.7155762
2: -506.6756592, 842.1623535, -567.6414185, 943.9479980, -1450.6235352, 1409.8037109
3: -869.5330811, 742.1264648, -977.8368530, 836.1256714, -1705.6585693, 1719.9633789
4: -634.5575562, 854.2334595, -713.3497925, 958.8959961, -1593.4536133, 1567.5832520

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_B2_A2_B2_A1_A1

### Relational analysis result of NS_B1_A1_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2097285, upper bound: 1339.2115945
time: 1.05 seconds

## Relational analysis of NS_B1_A1_B2_A2_B2_A1_A2

### Relational analysis result of NS_B1_A1_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2096757, upper bound: 1339.2112669
time: 0.89 seconds

## BFS NS instance: NS_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -279.8794250, 252.9188843, -279.8794250, 252.9188843, -532.7982788, 532.7982788
1: -1071.2863770, 914.2707520, -1071.2863770, 914.2707520, -1985.5571289, 1985.5571289
2: -567.6414185, 943.9479980, -567.6414185, 943.9479980, -1511.5893555, 1511.5893555
3: -977.8368530, 836.1256714, -977.8368530, 836.1256714, -1813.9625244, 1813.9625244
4: -713.3497925, 958.8959961, -713.3497925, 958.8959961, -1672.2458496, 1672.2458496

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2099148, upper bound: 1339.2116890
time: 2.60 seconds

## Relational analysis of NS_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2096757, upper bound: 1339.2116412
time: 1.05 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -259.8218384, 234.3891144, -236.7178955, 213.6591492, -473.4809875, 471.1069946
1: -993.4067383, 845.9050293, -904.4871826, 771.0987549, -1764.5054932, 1750.3919678
2: -528.0395508, 876.6255493, -480.5680542, 799.1917725, -1327.2313232, 1357.1936035
3: -907.6651611, 773.7341919, -826.8278198, 705.1502686, -1612.8150635, 1600.5617676
4: -662.1641846, 889.4735107, -603.4943237, 810.6299438, -1472.7941895, 1492.9676514

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_A2_B1_B1_A1_B1_A1

### Relational analysis result of NS_B1_A2_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2075410, upper bound: 1339.2096301
time: 1.73 seconds

## Relational analysis of NS_B1_A2_B1_B1_A1_B1_A2

### Relational analysis result of NS_B1_A2_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2075410, upper bound: 1339.2096301
time: 0.88 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -260.4254761, 235.0021667, -244.7930603, 221.1614990, -481.5869141, 479.7951965
1: -995.6641846, 848.2338867, -935.5700684, 797.7273560, -1793.3914795, 1783.8037109
2: -529.2808838, 878.5963745, -498.2113037, 827.8669434, -1357.1478271, 1376.8076172
3: -909.7088623, 775.6533203, -854.9373169, 729.5183105, -1639.2271729, 1630.5905762
4: -663.7349854, 891.5463257, -623.9547119, 839.7837524, -1503.5186768, 1515.5008545

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_A2_B1_B1_A1_B2_A1

### Relational analysis result of NS_B1_A2_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2075410, upper bound: 1339.2096301
time: 1.07 seconds

## Relational analysis of NS_B1_A2_B1_B1_A1_B2_A2

### Relational analysis result of NS_B1_A2_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2075410, upper bound: 1339.2096301
time: 1.20 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -289.1253052, 260.9322815, -237.1133575, 213.9381104, -503.0633850, 498.0456543
1: -1107.2126465, 943.1965942, -906.1936035, 772.2202148, -1879.4327393, 1849.3901367
2: -585.9553833, 972.5211182, -481.5543213, 800.2135010, -1386.1685791, 1454.0753174
3: -1010.4301147, 862.1947632, -827.9500122, 706.1353760, -1716.5654297, 1690.1445312
4: -737.1601562, 987.6393433, -604.3559570, 811.4989014, -1548.6590576, 1591.9953613

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_B1_B1_A2_B1_A1

### Relational analysis result of NS_B1_A2_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2107433, upper bound: 1339.2127397
time: 1.19 seconds

## Relational analysis of NS_B1_A2_B1_B1_A2_B1_A2

### Relational analysis result of NS_B1_A2_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2107433, upper bound: 1339.2133108
time: 0.89 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -289.7500000, 261.5377502, -235.3769531, 212.3244171, -502.0744019, 496.9147034
1: -1109.4555664, 945.6228027, -899.4669800, 766.7175903, -1876.1730957, 1845.0893555
2: -587.0828857, 974.7050781, -477.5150146, 793.8774414, -1380.9602051, 1452.2200928
3: -1012.6610107, 864.4035645, -822.1340942, 700.9407349, -1713.6014404, 1686.5374756
4: -738.7623901, 990.1739502, -600.1196899, 805.3220825, -1544.0844727, 1590.2937012

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_B1_B1_A2_B2_A1

### Relational analysis result of NS_B1_A2_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2115167, upper bound: 1339.2127993
time: 1.14 seconds

## Relational analysis of NS_B1_A2_B1_B1_A2_B2_A2

### Relational analysis result of NS_B1_A2_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2108849, upper bound: 1339.2133370
time: 0.80 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -259.3377991, 233.9734497, -271.8391724, 245.5500793, -504.8878784, 505.8126221
1: -991.4920044, 844.5764160, -1040.1838379, 887.8743286, -1879.3662109, 1884.7600098
2: -527.0584717, 874.6990967, -551.0593262, 915.8735352, -1442.9317627, 1425.7584229
3: -905.9906616, 772.2731934, -949.8130493, 811.6398315, -1717.6302490, 1722.0861816
4: -661.0531006, 887.6106567, -693.1937866, 930.1290283, -1591.1821289, 1580.8043213

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_B1_A2_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_A2_B1_B2_A1_A1_A1

### Relational analysis result of NS_B1_A2_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2114676, upper bound: 1339.2116354
time: 1.12 seconds

## Relational analysis of NS_B1_A2_B1_B2_A1_A1_A2

### Relational analysis result of NS_B1_A2_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2075000, upper bound: 1339.2101295
time: 0.95 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -289.5389709, 261.3972778, -271.8391724, 245.5500793, -535.0890503, 533.2364502
1: -1108.5106201, 945.0742798, -1040.1838379, 887.8743286, -1996.3850098, 1985.2579346
2: -586.8018188, 974.2145996, -551.0593262, 915.8735352, -1502.6751709, 1525.2739258
3: -1011.9727783, 863.8530273, -949.8130493, 811.6398315, -1823.6124268, 1813.6660156
4: -738.3388062, 989.6480103, -693.1937866, 930.1290283, -1668.4677734, 1682.8417969

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B1_A2_B1_B2_A1_A2_B1

### Relational analysis result of NS_B1_A2_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2075472, upper bound: 1339.2134111
time: 1.10 seconds

## Relational analysis of NS_B1_A2_B1_B2_A1_A2_B2

### Relational analysis result of NS_B1_A2_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2075000, upper bound: 1339.2119374
time: 0.96 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -326.5452881, 292.5143433, -271.8391724, 245.5500793, -572.0953369, 564.3535156
1: -1251.0766602, 1055.8413086, -1040.1838379, 887.8743286, -2138.9509277, 2096.0249023
2: -662.1367798, 1095.8020020, -551.0593262, 915.8735352, -1578.0101318, 1646.8613281
3: -1140.1081543, 967.6179810, -949.8130493, 811.6398315, -1951.7479248, 1917.4310303
4: -830.6295776, 1109.9207764, -693.1937866, 930.1290283, -1760.7585449, 1803.1145020

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B1_A2_B1_B2_A2_A1_B1

### Relational analysis result of NS_B1_A2_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2075901, upper bound: 1339.2110344
time: 1.09 seconds

## Relational analysis of NS_B1_A2_B1_B2_A2_A1_B2

### Relational analysis result of NS_B1_A2_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2075372, upper bound: 1339.2099557
time: 0.93 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -356.9777527, 320.1082458, -271.8391724, 245.5500793, -602.5278320, 591.9473877
1: -1369.1856689, 1156.6866455, -1040.1838379, 887.8743286, -2257.0593262, 2196.8703613
2: -722.6016235, 1195.6416016, -551.0593262, 915.8735352, -1638.4748535, 1746.7009277
3: -1247.0222168, 1059.7156982, -949.8130493, 811.6398315, -2058.6618652, 2009.5286865
4: -908.6473999, 1212.4528809, -693.1937866, 930.1290283, -1838.7762451, 1905.6467285

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_B1_B2_A2_A2_B1

### Relational analysis result of NS_B1_A2_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2123930, upper bound: 1339.2142816
time: 1.05 seconds

## Relational analysis of NS_B1_A2_B1_B2_A2_A2_B2

### Relational analysis result of NS_B1_A2_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2128120, upper bound: 1339.2142440
time: 0.99 seconds

## BFS NS instance: NS_B1_A2_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -259.3377991, 233.9734497, -307.4197388, 275.4576416, -534.7954102, 541.3931885
1: -991.4920044, 844.5764160, -1177.4262695, 994.2976074, -1985.7893066, 2022.0025635
2: -527.0584717, 874.6990967, -622.2865601, 1031.9949951, -1559.0533447, 1496.9855957
3: -905.9906616, 772.2731934, -1073.4550781, 911.1184082, -1817.1090088, 1845.7280273
4: -661.0531006, 887.6106567, -782.1300659, 1045.1027832, -1706.1558838, 1669.7407227

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_A2_B2_A1_A1_B1_A1

### Relational analysis result of NS_B1_A2_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2109795, upper bound: 1339.2101784
time: 1.35 seconds

## Relational analysis of NS_B1_A2_B2_A1_A1_B1_A2

### Relational analysis result of NS_B1_A2_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2072702, upper bound: 1339.2097127
time: 0.98 seconds

## BFS NS instance: NS_B1_A2_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -259.3377991, 233.9734497, -339.2242126, 304.3148193, -563.6525879, 573.1976318
1: -991.4920044, 844.5764160, -1300.6374512, 1099.3801270, -2090.8720703, 2145.2136230
2: -527.0584717, 874.6990967, -685.9866943, 1137.1968994, -1664.2551270, 1560.6857910
3: -905.9906616, 772.2731934, -1185.0489502, 1007.4972534, -1913.4875488, 1957.3221436
4: -661.0531006, 887.6106567, -863.5870361, 1153.0744629, -1814.1275635, 1751.1976318

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_B1_A2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_B2_A1_A1_B2_A1

### Relational analysis result of NS_B1_A2_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2084460, upper bound: 1339.2119751
time: 2.42 seconds

## Relational analysis of NS_B1_A2_B2_A1_A1_B2_A2

### Relational analysis result of NS_B1_A2_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2110307, upper bound: 1339.2144516
time: 0.96 seconds

## BFS NS instance: NS_B1_A2_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -289.5389709, 261.3972778, -307.4197388, 275.4576416, -564.9965820, 568.8170166
1: -1108.5106201, 945.0742798, -1177.4262695, 994.2976074, -2102.8081055, 2122.5002441
2: -586.8018188, 974.2145996, -622.2865601, 1031.9949951, -1618.7966309, 1596.5008545
3: -1011.9727783, 863.8530273, -1073.4550781, 911.1184082, -1923.0911865, 1937.3081055
4: -738.3388062, 989.6480103, -782.1300659, 1045.1027832, -1783.4414062, 1771.7780762

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_B1_A2_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_B2_A1_A2_B1_B1

### Relational analysis result of NS_B1_A2_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2135518, upper bound: 1339.2133419
time: 1.24 seconds

## Relational analysis of NS_B1_A2_B2_A1_A2_B1_B2

### Relational analysis result of NS_B1_A2_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2137733, upper bound: 1339.2133489
time: 1.07 seconds

## BFS NS instance: NS_B1_A2_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -289.5389709, 261.3972778, -339.2242126, 304.3148193, -593.8537598, 600.6213989
1: -1108.5106201, 945.0742798, -1300.6374512, 1099.3801270, -2207.8906250, 2245.7109375
2: -586.8018188, 974.2145996, -685.9866943, 1137.1968994, -1723.9982910, 1660.2009277
3: -1011.9727783, 863.8530273, -1185.0489502, 1007.4972534, -2019.4697266, 2048.9018555
4: -738.3388062, 989.6480103, -863.5870361, 1153.0744629, -1891.4130859, 1853.2351074

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_A2_B2_A1_A2_B2_A1

### Relational analysis result of NS_B1_A2_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2112524, upper bound: 1339.2121161
time: 0.99 seconds

## Relational analysis of NS_B1_A2_B2_A1_A2_B2_A2

### Relational analysis result of NS_B1_A2_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2091876, upper bound: 1339.2119120
time: 1.17 seconds

## BFS NS instance: NS_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -326.5452881, 292.5143433, -307.4197388, 275.4576416, -602.0029297, 599.9340820
1: -1251.0766602, 1055.8413086, -1177.4262695, 994.2976074, -2245.3742676, 2233.2673340
2: -662.1367798, 1095.8020020, -622.2865601, 1031.9949951, -1694.1315918, 1718.0882568
3: -1140.1081543, 967.6179810, -1073.4550781, 911.1184082, -2051.2265625, 2041.0729980
4: -830.6295776, 1109.9207764, -782.1300659, 1045.1027832, -1875.7322998, 1892.0507812

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2073930, upper bound: 1339.2110811
time: 0.96 seconds

## Relational analysis of NS_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2072633, upper bound: 1339.2095818
time: 1.11 seconds

## BFS NS instance: NS_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -356.9777527, 320.1082458, -307.4197388, 275.4576416, -632.4354248, 627.5279541
1: -1369.1856689, 1156.6866455, -1177.4262695, 994.2976074, -2363.4829102, 2334.1125488
2: -722.6016235, 1195.6416016, -622.2865601, 1031.9949951, -1754.5964355, 1817.9281006
3: -1247.0222168, 1059.7156982, -1073.4550781, 911.1184082, -2158.1406250, 2133.1704102
4: -908.6473999, 1212.4528809, -782.1300659, 1045.1027832, -1953.7497559, 1994.5830078

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2124109, upper bound: 1339.2132828
time: 0.95 seconds

## Relational analysis of NS_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2126845, upper bound: 1339.2133260
time: 0.87 seconds

## BFS NS instance: NS_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -326.5452881, 292.5143433, -339.2242126, 304.3148193, -630.8601074, 631.7385254
1: -1251.0766602, 1055.8413086, -1300.6374512, 1099.3801270, -2350.4567871, 2356.4785156
2: -662.1367798, 1095.8020020, -685.9866943, 1137.1968994, -1799.3333740, 1781.7883301
3: -1140.1081543, 967.6179810, -1185.0489502, 1007.4972534, -2147.6054688, 2152.6665039
4: -830.6295776, 1109.9207764, -863.5870361, 1153.0744629, -1983.7039795, 1973.5078125

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_B1_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_A2_B2_A2_B2_A1_A1

### Relational analysis result of NS_B1_A2_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2109757, upper bound: 1339.2114239
time: 0.94 seconds

## Relational analysis of NS_B1_A2_B2_A2_B2_A1_A2

### Relational analysis result of NS_B1_A2_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2072854, upper bound: 1339.2099556
time: 1.30 seconds

## BFS NS instance: NS_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -356.9777527, 320.1082458, -339.2242126, 304.3148193, -661.2926025, 659.3324585
1: -1369.1856689, 1156.6866455, -1300.6374512, 1099.3801270, -2468.5654297, 2457.3239746
2: -722.6016235, 1195.6416016, -685.9866943, 1137.1968994, -1859.7982178, 1881.6281738
3: -1247.0222168, 1059.7156982, -1185.0489502, 1007.4972534, -2254.5195312, 2244.7639160
4: -908.6473999, 1212.4528809, -863.5870361, 1153.0744629, -2061.7214355, 2076.0397949

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2073642, upper bound: 1339.2131338
time: 1.23 seconds

## Relational analysis of NS_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2072854, upper bound: 1339.2116369
time: 0.93 seconds

## BFS NS instance: NS_B2_A1_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -236.7178955, 213.6591492, -259.8218384, 234.3891144, -471.1069946, 473.4809875
1: -904.4871826, 771.0987549, -993.4067383, 845.9050293, -1750.3919678, 1764.5054932
2: -480.5680542, 799.1917725, -528.0395508, 876.6255493, -1357.1936035, 1327.2313232
3: -826.8278198, 705.1502686, -907.6651611, 773.7341919, -1600.5618896, 1612.8150635
4: -603.4943237, 810.6299438, -662.1641846, 889.4735107, -1492.9676514, 1472.7941895

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A1_A1_A1_B1_A1_B1

### Relational analysis result of NS_B2_A1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2096301, upper bound: 1339.2075410
time: 0.97 seconds

## Relational analysis of NS_B2_A1_A1_A1_B1_A1_B2

### Relational analysis result of NS_B2_A1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2096301, upper bound: 1339.2075410
time: 1.12 seconds

## BFS NS instance: NS_B2_A1_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -244.7930603, 221.1614990, -260.4254761, 235.0021667, -479.7951965, 481.5869446
1: -935.5700684, 797.7273560, -995.6641846, 848.2338867, -1783.8037109, 1793.3916016
2: -498.2113037, 827.8669434, -529.2808838, 878.5963745, -1376.8076172, 1357.1478271
3: -854.9373169, 729.5183105, -909.7088623, 775.6533203, -1630.5905762, 1639.2271729
4: -623.9547119, 839.7837524, -663.7349854, 891.5463257, -1515.5007324, 1503.5186768

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A1_A1_A1_B1_A2_B1

### Relational analysis result of NS_B2_A1_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2096301, upper bound: 1339.2075410
time: 0.94 seconds

## Relational analysis of NS_B2_A1_A1_A1_B1_A2_B2

### Relational analysis result of NS_B2_A1_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2096301, upper bound: 1339.2075410
time: 1.14 seconds

## BFS NS instance: NS_B2_A1_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -237.1133575, 213.9381104, -289.1253052, 260.9322815, -498.0456543, 503.0634155
1: -906.1936035, 772.2202148, -1107.2126465, 943.1965942, -1849.3901367, 1879.4328613
2: -481.5543213, 800.2135010, -585.9553833, 972.5211182, -1454.0753174, 1386.1685791
3: -827.9500122, 706.1353760, -1010.4301147, 862.1947632, -1690.1446533, 1716.5654297
4: -604.3559570, 811.4989014, -737.1601562, 987.6393433, -1591.9953613, 1548.6590576

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_A1_A1_B2_A1_B1

### Relational analysis result of NS_B2_A1_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2127397, upper bound: 1339.2113814
time: 9.04 seconds

## Relational analysis of NS_B2_A1_A1_A1_B2_A1_B2

### Relational analysis result of NS_B2_A1_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2127397, upper bound: 1339.2135892
time: 1.00 seconds

## BFS NS instance: NS_B2_A1_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -235.3769531, 212.3244171, -289.7500000, 261.5377502, -496.9147034, 502.0744019
1: -899.4669800, 766.7175903, -1109.4555664, 945.6228027, -1845.0893555, 1876.1730957
2: -477.5150146, 793.8774414, -587.0828857, 974.7050781, -1452.2198486, 1380.9600830
3: -822.1340942, 700.9407349, -1012.6610107, 864.4035645, -1686.5374756, 1713.6014404
4: -600.1196899, 805.3220825, -738.7623901, 990.1739502, -1590.2937012, 1544.0844727

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_A1_A1_B2_A2_B1

### Relational analysis result of NS_B2_A1_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2127993, upper bound: 1339.2115167
time: 1.04 seconds

## Relational analysis of NS_B2_A1_A1_A1_B2_A2_B2

### Relational analysis result of NS_B2_A1_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2127993, upper bound: 1339.2138730
time: 1.14 seconds

## BFS NS instance: NS_B2_A1_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -271.8391724, 245.5500793, -259.3377991, 233.9734497, -505.8126221, 504.8878784
1: -1040.1838379, 887.8743286, -991.4920044, 844.5764160, -1884.7600098, 1879.3660889
2: -551.0593262, 915.8735352, -527.0584717, 874.6990967, -1425.7584229, 1442.9318848
3: -949.8130493, 811.6398315, -905.9906616, 772.2731934, -1722.0860596, 1717.6303711
4: -693.1937866, 930.1290283, -661.0531006, 887.6106567, -1580.8043213, 1591.1821289

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_B2_A1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B2_A1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A1_A1_A2_B1_B1_B1

### Relational analysis result of NS_B2_A1_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2102127, upper bound: 1339.2114676
time: 0.98 seconds

## Relational analysis of NS_B2_A1_A1_A2_B1_B1_B2

### Relational analysis result of NS_B2_A1_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2101295, upper bound: 1339.2075000
time: 0.98 seconds

## BFS NS instance: NS_B2_A1_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -271.8391724, 245.5500793, -289.5389709, 261.3972778, -533.2364502, 535.0890503
1: -1040.1838379, 887.8743286, -1108.5106201, 945.0742798, -1985.2579346, 1996.3850098
2: -551.0593262, 915.8735352, -586.8018188, 974.2145996, -1525.2738037, 1502.6750488
3: -949.8130493, 811.6398315, -1011.9727783, 863.8530273, -1813.6660156, 1823.6125488
4: -693.1937866, 930.1290283, -738.3388062, 989.6480103, -1682.8417969, 1668.4677734

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B2_A1_A1_A2_B1_B2_A1

### Relational analysis result of NS_B2_A1_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2113185, upper bound: 1339.2095022
time: 1.29 seconds

## Relational analysis of NS_B2_A1_A1_A2_B1_B2_A2

### Relational analysis result of NS_B2_A1_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2101295, upper bound: 1339.2094262
time: 0.98 seconds

## BFS NS instance: NS_B2_A1_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -271.8391724, 245.5500793, -326.5452881, 292.5143433, -564.3535156, 572.0953369
1: -1040.1838379, 887.8743286, -1251.0766602, 1055.8413086, -2096.0246582, 2138.9509277
2: -551.0593262, 915.8735352, -662.1367798, 1095.8020020, -1646.8613281, 1578.0100098
3: -949.8130493, 811.6398315, -1140.1081543, 967.6179810, -1917.4310303, 1951.7480469
4: -693.1937866, 930.1290283, -830.6295776, 1109.9207764, -1803.1145020, 1760.7585449

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B2_A1_A1_A2_B2_B1_A1

### Relational analysis result of NS_B2_A1_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2110344, upper bound: 1339.2075901
time: 0.96 seconds

## Relational analysis of NS_B2_A1_A1_A2_B2_B1_A2

### Relational analysis result of NS_B2_A1_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2099557, upper bound: 1339.2075372
time: 0.91 seconds

## BFS NS instance: NS_B2_A1_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -271.8391724, 245.5500793, -356.9777527, 320.1082458, -591.9473877, 602.5278320
1: -1040.1838379, 887.8743286, -1369.1856689, 1156.6866455, -2196.8703613, 2257.0593262
2: -551.0593262, 915.8735352, -722.6016235, 1195.6416016, -1746.7009277, 1638.4749756
3: -949.8130493, 811.6398315, -1247.0222168, 1059.7156982, -2009.5288086, 2058.6621094
4: -693.1937866, 930.1290283, -908.6473999, 1212.4528809, -1905.6467285, 1838.7762451

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_A1_A2_B2_B2_A1

### Relational analysis result of NS_B2_A1_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2143336, upper bound: 1339.2132152
time: 1.01 seconds

## Relational analysis of NS_B2_A1_A1_A2_B2_B2_A2

### Relational analysis result of NS_B2_A1_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2144132, upper bound: 1339.2136926
time: 1.03 seconds

## BFS NS instance: NS_B2_A1_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -307.4197388, 275.4576416, -259.3377991, 233.9734497, -541.3931885, 534.7954102
1: -1177.4262695, 994.2976074, -991.4920044, 844.5764160, -2022.0025635, 1985.7893066
2: -622.2865601, 1031.9949951, -527.0584717, 874.6990967, -1496.9855957, 1559.0533447
3: -1073.4550781, 911.1184082, -905.9906616, 772.2731934, -1845.7281494, 1817.1090088
4: -782.1300659, 1045.1027832, -661.0531006, 887.6106567, -1669.7407227, 1706.1558838

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A1_A2_B1_B1_A1_B1

### Relational analysis result of NS_B2_A1_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2101784, upper bound: 1339.2109795
time: 1.09 seconds

## Relational analysis of NS_B2_A1_A2_B1_B1_A1_B2

### Relational analysis result of NS_B2_A1_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2097127, upper bound: 1339.2072702
time: 0.92 seconds

## BFS NS instance: NS_B2_A1_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -339.2242126, 304.3148193, -259.3377991, 233.9734497, -573.1976318, 563.6525879
1: -1300.6374512, 1099.3801270, -991.4920044, 844.5764160, -2145.2136230, 2090.8720703
2: -685.9866943, 1137.1968994, -527.0584717, 874.6990967, -1560.6857910, 1664.2551270
3: -1185.0489502, 1007.4972534, -905.9906616, 772.2731934, -1957.3220215, 1913.4875488
4: -863.5870361, 1153.0744629, -661.0531006, 887.6106567, -1751.1976318, 1814.1275635

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_B2_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B2_A1_A2_B1_B1_A2_B1

### Relational analysis result of NS_B2_A1_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2109018, upper bound: 1339.2084460
time: 0.92 seconds

## Relational analysis of NS_B2_A1_A2_B1_B1_A2_B2

### Relational analysis result of NS_B2_A1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2115840, upper bound: 1339.2110307
time: 1.00 seconds

## BFS NS instance: NS_B2_A1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -307.4197388, 275.4576416, -289.5389709, 261.3972778, -568.8170166, 564.9965820
1: -1177.4262695, 994.2976074, -1108.5106201, 945.0742798, -2122.5002441, 2102.8081055
2: -622.2865601, 1031.9949951, -586.8018188, 974.2145996, -1596.5009766, 1618.7966309
3: -1073.4550781, 911.1184082, -1011.9727783, 863.8530273, -1937.3081055, 1923.0911865
4: -782.1300659, 1045.1027832, -738.3388062, 989.6480103, -1771.7780762, 1783.4414062

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_B2_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_A2_B1_B2_A1_A1

### Relational analysis result of NS_B2_A1_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2133419, upper bound: 1339.2135518
time: 1.03 seconds

## Relational analysis of NS_B2_A1_A2_B1_B2_A1_A2

### Relational analysis result of NS_B2_A1_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2133489, upper bound: 1339.2137733
time: 0.89 seconds

## BFS NS instance: NS_B2_A1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -339.2242126, 304.3148193, -289.5389709, 261.3972778, -600.6213989, 593.8537598
1: -1300.6374512, 1099.3801270, -1108.5106201, 945.0742798, -2245.7109375, 2207.8906250
2: -685.9866943, 1137.1968994, -586.8018188, 974.2145996, -1660.2010498, 1723.9982910
3: -1185.0489502, 1007.4972534, -1011.9727783, 863.8530273, -2048.9018555, 2019.4697266
4: -863.5870361, 1153.0744629, -738.3388062, 989.6480103, -1853.2351074, 1891.4130859

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A1_A2_B1_B2_A2_B1

### Relational analysis result of NS_B2_A1_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2101784, upper bound: 1339.2126534
time: 1.03 seconds

## Relational analysis of NS_B2_A1_A2_B1_B2_A2_B2

### Relational analysis result of NS_B2_A1_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2097127, upper bound: 1339.2090968
time: 0.92 seconds

## BFS NS instance: NS_B2_A1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -307.4197388, 275.4576416, -326.5452881, 292.5143433, -599.9340820, 602.0029297
1: -1177.4262695, 994.2976074, -1251.0766602, 1055.8413086, -2233.2673340, 2245.3742676
2: -622.2865601, 1031.9949951, -662.1367798, 1095.8020020, -1718.0882568, 1694.1315918
3: -1073.4550781, 911.1184082, -1140.1081543, 967.6179810, -2041.0729980, 2051.2265625
4: -782.1300659, 1045.1027832, -830.6295776, 1109.9207764, -1892.0507812, 1875.7322998

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B2_A1_A2_B2_A1_B1_A1

### Relational analysis result of NS_B2_A1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2110811, upper bound: 1339.2073794
time: 0.80 seconds

## Relational analysis of NS_B2_A1_A2_B2_A1_B1_A2

### Relational analysis result of NS_B2_A1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2095818, upper bound: 1339.2072719
time: 1.02 seconds

## BFS NS instance: NS_B2_A1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -307.4197388, 275.4576416, -356.9777527, 320.1082458, -627.5279541, 632.4354248
1: -1177.4262695, 994.2976074, -1369.1856689, 1156.6866455, -2334.1125488, 2363.4829102
2: -622.2865601, 1031.9949951, -722.6016235, 1195.6416016, -1817.9281006, 1754.5964355
3: -1073.4550781, 911.1184082, -1247.0222168, 1059.7156982, -2133.1704102, 2158.1406250
4: -782.1300659, 1045.1027832, -908.6473999, 1212.4528809, -1994.5830078, 1953.7497559

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_B2_A1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_A2_B2_A1_B2_A1

### Relational analysis result of NS_B2_A1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2132828, upper bound: 1339.2135518
time: 1.07 seconds

## Relational analysis of NS_B2_A1_A2_B2_A1_B2_A2

### Relational analysis result of NS_B2_A1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2133260, upper bound: 1339.2137733
time: 0.92 seconds

## BFS NS instance: NS_B2_A1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -339.2242126, 304.3148193, -326.5452881, 292.5143433, -631.7385254, 630.8601074
1: -1300.6374512, 1099.3801270, -1251.0766602, 1055.8413086, -2356.4782715, 2350.4567871
2: -685.9866943, 1137.1968994, -662.1367798, 1095.8020020, -1781.7883301, 1799.3333740
3: -1185.0489502, 1007.4972534, -1140.1081543, 967.6179810, -2152.6665039, 2147.6054688
4: -863.5870361, 1153.0744629, -830.6295776, 1109.9207764, -1973.5078125, 1983.7039795

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_B2_A1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B2_A1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A1_A2_B2_A2_B1_B1

### Relational analysis result of NS_B2_A1_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2114239, upper bound: 1339.2109757
time: 1.18 seconds

## Relational analysis of NS_B2_A1_A2_B2_A2_B1_B2

### Relational analysis result of NS_B2_A1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2099556, upper bound: 1339.2072719
time: 1.03 seconds

## BFS NS instance: NS_B2_A1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -339.2242126, 304.3148193, -356.9777527, 320.1082458, -659.3324585, 661.2926025
1: -1300.6374512, 1099.3801270, -1369.1856689, 1156.6866455, -2457.3239746, 2468.5654297
2: -685.9866943, 1137.1968994, -722.6016235, 1195.6416016, -1881.6281738, 1859.7982178
3: -1185.0489502, 1007.4972534, -1247.0222168, 1059.7156982, -2244.7644043, 2254.5195312
4: -863.5870361, 1153.0744629, -908.6473999, 1212.4528809, -2076.0397949, 2061.7216797

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B2_A1_A2_B2_A2_B2_A1

### Relational analysis result of NS_B2_A1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2110276, upper bound: 1339.2092780
time: 1.01 seconds

## Relational analysis of NS_B2_A1_A2_B2_A2_B2_A2

### Relational analysis result of NS_B2_A1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2099556, upper bound: 1339.2090920
time: 1.12 seconds

## BFS NS instance: NS_B2_A2_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -259.6002808, 234.1820068, -259.6002808, 234.1820068, -493.7821655, 493.7821655
1: -992.5734863, 845.0814209, -992.5734863, 845.0814209, -1837.6546631, 1837.6546631
2: -527.6042480, 876.0423584, -527.6042480, 876.0423584, -1403.6466064, 1403.6466064
3: -906.8853760, 773.0839233, -906.8853760, 773.0839233, -1679.9692383, 1679.9692383
4: -661.5457764, 888.8338623, -661.5457764, 888.8338623, -1550.3795166, 1550.3795166

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A1_B1_B1_A1_A1

### Relational analysis result of NS_B2_A2_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2126930, upper bound: 1339.2129176
time: 1.09 seconds

## Relational analysis of NS_B2_A2_A1_B1_B1_A1_A2

### Relational analysis result of NS_B2_A2_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2126930, upper bound: 1339.2127875
time: 0.87 seconds

## BFS NS instance: NS_B2_A2_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -289.9319153, 261.7218933, -259.6002808, 234.1820068, -524.1138306, 521.3221436
1: -1110.0667725, 946.1136475, -992.5734863, 845.0814209, -1955.1478271, 1938.6870117
2: -587.6096802, 975.8930054, -527.6042480, 876.0423584, -1463.6520996, 1503.4971924
3: -1013.3492432, 865.1301270, -906.8853760, 773.0839233, -1786.4331055, 1772.0155029
4: -739.1745605, 991.2822876, -661.5457764, 888.8338623, -1628.0083008, 1652.8278809

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A1_B1_B1_A2_B1

### Relational analysis result of NS_B2_A2_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2128453, upper bound: 1339.2127875
time: 0.91 seconds

## Relational analysis of NS_B2_A2_A1_B1_B1_A2_B2

### Relational analysis result of NS_B2_A2_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2126930, upper bound: 1339.2127875
time: 0.91 seconds

## BFS NS instance: NS_B2_A2_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -259.6002808, 234.1820068, -289.9319153, 261.7218933, -521.3220825, 524.1138306
1: -992.5734863, 845.0814209, -1110.0667725, 946.1136475, -1938.6870117, 1955.1479492
2: -527.6042480, 876.0423584, -587.6096802, 975.8930054, -1503.4971924, 1463.6520996
3: -906.8853760, 773.0839233, -1013.3492432, 865.1301270, -1772.0155029, 1786.4331055
4: -661.5457764, 888.8338623, -739.1745605, 991.2822876, -1652.8280029, 1628.0081787

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A1_B1_B2_A1_A1

### Relational analysis result of NS_B2_A2_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2126911, upper bound: 1339.2138730
time: 1.10 seconds

## Relational analysis of NS_B2_A2_A1_B1_B2_A1_A2

### Relational analysis result of NS_B2_A2_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2126911, upper bound: 1339.2137936
time: 1.16 seconds

## BFS NS instance: NS_B2_A2_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -289.9319153, 261.7218933, -289.9319153, 261.7218933, -551.6536865, 551.6536865
1: -1110.0667725, 946.1136475, -1110.0667725, 946.1136475, -2056.1804199, 2056.1801758
2: -587.6096802, 975.8930054, -587.6096802, 975.8930054, -1563.5026855, 1563.5026855
3: -1013.3492432, 865.1301270, -1013.3492432, 865.1301270, -1878.4793701, 1878.4793701
4: -739.1745605, 991.2822876, -739.1745605, 991.2822876, -1730.4565430, 1730.4564209

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A2_A1_B1_B2_A2_B1

### Relational analysis result of NS_B2_A2_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2128453, upper bound: 1339.2134563
time: 1.11 seconds

## Relational analysis of NS_B2_A2_A1_B1_B2_A2_B2

### Relational analysis result of NS_B2_A2_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2126911, upper bound: 1339.2134581
time: 1.11 seconds

## BFS NS instance: NS_B2_A2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -264.2924500, 238.5483704, -268.9626160, 243.1861115, -507.4785156, 507.5109863
1: -1009.9717407, 861.3646240, -1027.9147949, 877.8071899, -1887.7789307, 1889.2794189
2: -536.3375244, 892.6682129, -546.9664307, 909.6557007, -1445.9931641, 1439.6346436
3: -923.5935059, 787.9431152, -939.8849487, 802.6157227, -1726.2092285, 1727.8280029
4: -673.6499634, 905.9495850, -685.6069336, 923.4005127, -1597.0505371, 1591.5565186

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_A1_B2_A1_B1_A1

### Relational analysis result of NS_B2_A2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2108718, upper bound: 1339.2096544
time: 1.11 seconds

## Relational analysis of NS_B2_A2_A1_B2_A1_B1_A2

### Relational analysis result of NS_B2_A2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2125895, upper bound: 1339.2098452
time: 1.20 seconds

## BFS NS instance: NS_B2_A2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -264.2924500, 238.5483704, -335.2837524, 301.0312805, -565.3236694, 573.8320923
1: -1009.9717407, 861.3646240, -1284.0396729, 1086.5528564, -2096.5246582, 2145.4040527
2: -536.3375244, 892.6682129, -679.7056274, 1127.6051025, -1663.9425049, 1572.3737793
3: -923.5935059, 787.9431152, -1171.0661621, 995.4823608, -1919.0759277, 1959.0090332
4: -673.6499634, 905.9495850, -852.9486084, 1142.9920654, -1816.6419678, 1758.8981934

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_A1_B2_A1_B2_A1

### Relational analysis result of NS_B2_A2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2108718, upper bound: 1339.2096544
time: 1.00 seconds

## Relational analysis of NS_B2_A2_A1_B2_A1_B2_A2

### Relational analysis result of NS_B2_A2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2125895, upper bound: 1339.2098452
time: 0.96 seconds

## BFS NS instance: NS_B2_A2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -331.0111084, 296.6192322, -268.9626160, 243.1861115, -574.1971436, 565.5818481
1: -1267.8787842, 1071.0562744, -1027.9147949, 877.8071899, -2145.6860352, 2098.9711914
2: -670.4309082, 1111.8437500, -546.9664307, 909.6557007, -1580.0863037, 1658.8101807
3: -1156.1138916, 981.8455200, -939.8849487, 802.6157227, -1958.7296143, 1921.7304688
4: -841.9860840, 1126.4869385, -685.6069336, 923.4005127, -1765.3865967, 1812.0938721

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_A1_B2_A2_B1_B1

### Relational analysis result of NS_B2_A2_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2108458, upper bound: 1339.2073756
time: 1.05 seconds

## Relational analysis of NS_B2_A2_A1_B2_A2_B1_B2

### Relational analysis result of NS_B2_A2_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2125895, upper bound: 1339.2097656
time: 0.98 seconds

## BFS NS instance: NS_B2_A2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -331.0111084, 296.6192322, -335.2837524, 301.0312805, -632.0423584, 631.9029541
1: -1267.8787842, 1071.0562744, -1284.0396729, 1086.5528564, -2354.4316406, 2355.0959473
2: -670.4309082, 1111.8437500, -679.7056274, 1127.6051025, -1798.0356445, 1791.5493164
3: -1156.1138916, 981.8455200, -1171.0661621, 995.4823608, -2151.5961914, 2152.9116211
4: -841.9860840, 1126.4869385, -852.9486084, 1142.9920654, -1984.9781494, 1979.4355469

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_A1_B2_A2_B2_A1

### Relational analysis result of NS_B2_A2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2108036, upper bound: 1339.2095491
time: 1.14 seconds

## Relational analysis of NS_B2_A2_A1_B2_A2_B2_A2

### Relational analysis result of NS_B2_A2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2125895, upper bound: 1339.2097656
time: 1.19 seconds

## BFS NS instance: NS_B2_A2_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -268.9626160, 243.1861115, -264.2924500, 238.5483704, -507.5109863, 507.4785156
1: -1027.9147949, 877.8071899, -1009.9717407, 861.3646240, -1889.2794189, 1887.7789307
2: -546.9664307, 909.6557007, -536.3375244, 892.6682129, -1439.6346436, 1445.9931641
3: -939.8849487, 802.6157227, -923.5935059, 787.9431152, -1727.8280029, 1726.2092285
4: -685.6069336, 923.4005127, -673.6499634, 905.9495850, -1591.5565186, 1597.0505371

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_A2_B1_B1_A1_B1

### Relational analysis result of NS_B2_A2_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2096544, upper bound: 1339.2108718
time: 1.12 seconds

## Relational analysis of NS_B2_A2_A2_B1_B1_A1_B2

### Relational analysis result of NS_B2_A2_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2098424, upper bound: 1339.2125895
time: 0.99 seconds

## BFS NS instance: NS_B2_A2_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -335.2837524, 301.0312805, -264.2924500, 238.5483704, -573.8320923, 565.3236694
1: -1284.0396729, 1086.5528564, -1009.9717407, 861.3646240, -2145.4042969, 2096.5246582
2: -679.7056274, 1127.6051025, -536.3375244, 892.6682129, -1572.3737793, 1663.9425049
3: -1171.0661621, 995.4823608, -923.5935059, 787.9431152, -1959.0091553, 1919.0759277
4: -852.9486084, 1142.9920654, -673.6499634, 905.9495850, -1758.8981934, 1816.6419678

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_A2_B1_B1_A2_B1

### Relational analysis result of NS_B2_A2_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2096544, upper bound: 1339.2108718
time: 0.90 seconds

## Relational analysis of NS_B2_A2_A2_B1_B1_A2_B2

### Relational analysis result of NS_B2_A2_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2098424, upper bound: 1339.2125895
time: 1.05 seconds

## BFS NS instance: NS_B2_A2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -268.9626160, 243.1861115, -331.0111084, 296.6192322, -565.5818481, 574.1971436
1: -1027.9147949, 877.8071899, -1267.8787842, 1071.0562744, -2098.9709473, 2145.6860352
2: -546.9664307, 909.6557007, -670.4309082, 1111.8437500, -1658.8101807, 1580.0863037
3: -939.8849487, 802.6157227, -1156.1138916, 981.8455200, -1921.7304688, 1958.7296143
4: -685.6069336, 923.4005127, -841.9860840, 1126.4869385, -1812.0938721, 1765.3865967

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_A2_B1_B2_A1_A1

### Relational analysis result of NS_B2_A2_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2073690, upper bound: 1339.2108458
time: 1.28 seconds

## Relational analysis of NS_B2_A2_A2_B1_B2_A1_A2

### Relational analysis result of NS_B2_A2_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2073642, upper bound: 1339.2125895
time: 1.30 seconds

## BFS NS instance: NS_B2_A2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -335.2837524, 301.0312805, -331.0111084, 296.6192322, -631.9029541, 632.0423584
1: -1284.0396729, 1086.5528564, -1267.8787842, 1071.0562744, -2355.0957031, 2354.4316406
2: -679.7056274, 1127.6051025, -670.4309082, 1111.8437500, -1791.5493164, 1798.0356445
3: -1171.0661621, 995.4823608, -1156.1138916, 981.8455200, -2152.9116211, 2151.5961914
4: -852.9486084, 1142.9920654, -841.9860840, 1126.4869385, -1979.4355469, 1984.9781494

Time for backsubstitution: 1.21 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.47 + 417.06 = 420.53 seconds
