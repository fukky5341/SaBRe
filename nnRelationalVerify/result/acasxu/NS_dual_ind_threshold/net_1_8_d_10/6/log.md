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
execution time: IAR + RelationalAnalysis = 1.04 + 2.32 = 3.36 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -1339.2157786, upper bound: 1339.2157786

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2157786, upper bound: 1339.2147286
time: 1.08 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2147286, upper bound: 1339.2147286
time: 1.00 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.17 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.17
Output dim: 3, lower bound: -1339.2157786, upper bound: 1339.2147286
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.17
Output dim: 3, lower bound: -1339.2147286, upper bound: 1339.2147286

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -253.9733124, 229.4320374, -256.1399231, 231.4644623, -485.4377441, 485.5719604
1: -970.0575562, 828.4106445, -978.1137085, 835.6396484, -1805.6972656, 1806.5242920
2: -515.2947388, 858.0770874, -519.8786011, 865.7119141, -1381.0065918, 1377.9555664
3: -887.2174072, 757.4891968, -894.7660522, 764.0308838, -1651.2482910, 1652.2552490
4: -647.4416504, 870.7501831, -653.0750122, 878.5214844, -1525.9630127, 1523.8251953

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2144893, upper bound: 1339.2129176
time: 1.31 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2144893, upper bound: 1339.2138820
time: 1.01 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -272.3238220, 245.8114471, -257.2024536, 232.5796967, -504.9035034, 503.0138245
1: -1040.8454590, 887.8678589, -981.9048462, 839.5596924, -1880.4051514, 1869.7724609
2: -552.4603271, 919.3123779, -522.2640381, 869.8874512, -1422.3477783, 1441.5761719
3: -951.5145874, 811.9313965, -898.4796143, 767.5150146, -1719.0294189, 1710.4107666
4: -694.0728760, 933.0305786, -655.9251099, 882.8465576, -1576.9193115, 1588.9556885

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2138730, upper bound: 1339.2129176
time: 1.77 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2138820, upper bound: 1339.2138820
time: 1.18 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.01 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.01
Output dim: 3, lower bound: -1339.2144893, upper bound: 1339.2129176
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.01
Output dim: 3, lower bound: -1339.2144893, upper bound: 1339.2138820
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.01
Output dim: 3, lower bound: -1339.2138730, upper bound: 1339.2129176
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.01
Output dim: 3, lower bound: -1339.2138820, upper bound: 1339.2138820

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -247.7953033, 223.7384644, -246.5203247, 222.5613251, -470.3565674, 470.2587891
1: -946.7644043, 807.9193115, -941.8807373, 803.4702759, -1750.2346191, 1749.8000488
2: -502.7835999, 836.5459595, -500.3629150, 832.0545044, -1334.8380127, 1336.9089355
3: -865.4487305, 738.6900024, -860.8981934, 734.5314941, -1599.9802246, 1599.5878906
4: -631.6959839, 848.7247925, -628.5392456, 844.1084595, -1475.8041992, 1477.2640381

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2113185, upper bound: 1339.2075986
time: 1.26 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2101295, upper bound: 1339.2075410
time: 1.06 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -248.4915924, 224.5117188, -278.2328186, 251.3709869, -499.8625793, 502.7445374
1: -949.1719360, 810.6821899, -1064.5222168, 908.7963867, -1857.9680176, 1875.2043457
2: -504.2864685, 839.2608032, -564.1215820, 937.6348267, -1441.9210205, 1403.3822021
3: -868.2354126, 741.0051270, -972.0776978, 830.7824097, -1699.0177002, 1713.0827637
4: -633.7103271, 851.7661133, -709.4787598, 952.3140869, -1586.0244141, 1561.2447510

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2134111, upper bound: 1339.2098565
time: 1.11 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2101295, upper bound: 1339.2096605
time: 0.91 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -266.4650574, 240.4792938, -247.4911957, 223.4970551, -489.9620972, 487.9704895
1: -1018.7454834, 868.3457031, -945.2848511, 807.0170288, -1825.7624512, 1813.6306152
2: -541.1629028, 898.9877930, -502.4352722, 835.6378174, -1376.8005371, 1401.4228516
3: -930.9078369, 794.0358887, -864.3301392, 737.6907959, -1668.5985107, 1658.3659668
4: -679.1300049, 912.3315430, -631.1503296, 847.8325195, -1526.9625244, 1543.4819336

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2113009, upper bound: 1339.2075986
time: 1.06 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2091784, upper bound: 1339.2075410
time: 1.01 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -266.8143005, 240.8507080, -279.1507263, 252.2880402, -519.1023560, 520.0013428
1: -1019.8461304, 869.9521484, -1067.7082520, 912.1237183, -1931.9696045, 1937.6604004
2: -541.3326416, 900.1993408, -566.0787964, 941.1123047, -1482.4449463, 1466.2780762
3: -932.4186401, 795.2553101, -975.3186646, 833.7177734, -1766.1363525, 1770.5738525
4: -680.2758789, 913.7545776, -711.9122925, 956.0179443, -1636.2933350, 1625.6668701

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2130148, upper bound: 1339.2098588
time: 1.23 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2096605, upper bound: 1339.2096605
time: 1.03 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.33 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 3, lower bound: -1339.2113185, upper bound: 1339.2075986
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 3, lower bound: -1339.2101295, upper bound: 1339.2075410
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 3, lower bound: -1339.2134111, upper bound: 1339.2098565
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 3, lower bound: -1339.2101295, upper bound: 1339.2096605
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 3, lower bound: -1339.2113009, upper bound: 1339.2075986
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 3, lower bound: -1339.2091784, upper bound: 1339.2075410
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 3, lower bound: -1339.2130148, upper bound: 1339.2098588
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 3, lower bound: -1339.2096605, upper bound: 1339.2096605

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -244.2470093, 220.5315552, -244.1975708, 220.4571686, -464.7041626, 464.7290955
1: -933.1945190, 796.0230713, -933.0048218, 795.6547852, -1728.8492432, 1729.0278320
2: -495.6360779, 824.9326172, -495.6884766, 824.4546509, -1320.0905762, 1320.6207275
3: -853.1392212, 728.0506592, -852.8425293, 727.5626221, -1580.7016602, 1580.8931885
4: -622.5960693, 836.8617554, -622.5761719, 836.3248901, -1458.9207764, 1459.4376221

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2051622, upper bound: 1339.2024900
time: 0.97 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2099840, upper bound: 1339.2051931
time: 1.14 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -252.2573853, 227.9869080, -244.4859314, 220.7602692, -473.0176392, 472.4728394
1: -964.0801392, 822.5290527, -934.0166626, 796.9324951, -1761.0126953, 1756.5456543
2: -513.0211792, 853.1254883, -496.3360291, 825.4848633, -1338.5061035, 1349.4615479
3: -881.0199585, 752.3135376, -853.7407837, 728.5869751, -1609.6069336, 1606.0543213
4: -642.9034424, 865.6689453, -623.3341675, 837.4296265, -1480.3330078, 1489.0030518

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2050664, upper bound: 1339.2024888
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2090605, upper bound: 1339.2051815
time: 1.08 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -244.7990570, 221.1696320, -276.0571594, 249.3775330, -494.1765442, 497.2267456
1: -935.0347900, 798.3237305, -1056.2159424, 901.5120850, -1836.5463867, 1854.5396729
2: -496.8507690, 827.1917725, -559.7429810, 930.5066528, -1427.3572998, 1386.9346924
3: -855.4066162, 729.9403687, -964.5451660, 824.2763062, -1679.6828613, 1694.4855957
4: -624.2350464, 839.4254150, -703.9028320, 945.0236816, -1569.2587891, 1543.3278809

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2127886, upper bound: 1339.2095789
time: 1.00 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2133936, upper bound: 1339.2098256
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -252.0183716, 227.6892090, -276.0691528, 249.4479523, -501.4663086, 503.7583313
1: -962.9070435, 822.0161133, -1056.1947021, 901.8345947, -1864.7415771, 1878.2108154
2: -511.6142883, 851.9074707, -559.8436279, 930.7031860, -1442.3173828, 1411.7508545
3: -880.5345459, 751.6891479, -964.4866333, 824.4559326, -1704.9902344, 1716.1757812
4: -642.5473633, 864.6139526, -703.9147949, 945.2479248, -1587.7949219, 1568.5285645

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2102846, upper bound: 1339.2094076
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2107886, upper bound: 1339.2096560
time: 1.11 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -262.5783997, 236.9383545, -245.1445923, 221.3850098, -483.9633789, 482.0829163
1: -1003.8460693, 855.3237305, -936.3123169, 799.1735840, -1803.0196533, 1791.6359863
2: -533.3724976, 886.3398438, -497.7159119, 828.0154419, -1361.3879395, 1384.0557861
3: -917.3871460, 782.4293823, -856.1762085, 730.7075806, -1648.0947266, 1638.6055908
4: -669.1592407, 899.3970947, -625.1149902, 840.0286255, -1509.1878662, 1524.5117188

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2052727, upper bound: 1339.2024900
time: 0.98 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2098123, upper bound: 1339.2051931
time: 1.26 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -267.4748840, 241.8156433, -245.3094330, 221.5593262, -489.0342102, 487.1250000
1: -1022.6122437, 872.5455933, -936.8479004, 799.9979248, -1822.6098633, 1809.3933105
2: -544.4107666, 904.4436646, -498.1073608, 828.5407715, -1372.9515381, 1402.5510254
3: -934.3807373, 797.7935181, -856.6636963, 731.2996826, -1665.6802979, 1654.4570312
4: -681.6704712, 917.8596191, -625.5781860, 840.6203613, -1522.2905273, 1543.4377441

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2050679, upper bound: 1339.2024888
time: 1.06 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2050679, upper bound: 1339.2051815
time: 1.32 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -262.8648376, 237.2611847, -276.9076843, 250.2458496, -513.1107178, 514.1688843
1: -1004.7023926, 856.7432251, -1059.1340332, 904.6642456, -1909.3666992, 1915.8771973
2: -533.4082031, 887.3182373, -561.5630493, 933.7899780, -1467.1981201, 1448.8812256
3: -918.6716919, 783.4802856, -967.5471191, 827.0424805, -1745.7141113, 1751.0272217
4: -670.1300049, 900.5930786, -706.1663818, 948.5296631, -1618.6596680, 1606.7591553

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2125892, upper bound: 1339.2095858
time: 0.99 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2129888, upper bound: 1339.2098385
time: 1.11 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -267.0506287, 241.3401947, -276.7843628, 250.1826782, -517.2332764, 518.1245117
1: -1020.7564087, 871.5045166, -1058.5966797, 904.5018311, -1925.2580566, 1930.1011963
2: -542.2412720, 902.3704224, -561.3840942, 933.4649048, -1475.7061768, 1463.7545166
3: -933.2692261, 796.6746216, -967.0192261, 826.7858887, -1760.0549316, 1763.6938477
4: -680.8818970, 915.9735718, -705.8363037, 948.2483521, -1629.1300049, 1621.8098145

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2094263, upper bound: 1339.2094076
time: 1.15 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2096560, upper bound: 1339.2096560
time: 1.09 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.67 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 3, lower bound: -1339.2051622, upper bound: 1339.2024900
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 3, lower bound: -1339.2099840, upper bound: 1339.2051931
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 3, lower bound: -1339.2050664, upper bound: 1339.2024888
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 3, lower bound: -1339.2090605, upper bound: 1339.2051815
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 3, lower bound: -1339.2127886, upper bound: 1339.2095789
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 3, lower bound: -1339.2133936, upper bound: 1339.2098256
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 3, lower bound: -1339.2102846, upper bound: 1339.2094076
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 3, lower bound: -1339.2107886, upper bound: 1339.2096560
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 3, lower bound: -1339.2052727, upper bound: 1339.2024900
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 3, lower bound: -1339.2098123, upper bound: 1339.2051931
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 3, lower bound: -1339.2050679, upper bound: 1339.2024888
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 3, lower bound: -1339.2050679, upper bound: 1339.2051815
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 3, lower bound: -1339.2125892, upper bound: 1339.2095858
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 3, lower bound: -1339.2129888, upper bound: 1339.2098385
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 3, lower bound: -1339.2094263, upper bound: 1339.2094076
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 3, lower bound: -1339.2096560, upper bound: 1339.2096560

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -240.7051239, 217.3187714, -239.3999329, 216.0862427, -456.7913818, 456.7186890
1: -920.0797119, 784.2710571, -915.2236938, 779.5722656, -1699.6518555, 1699.4945068
2: -488.4908142, 812.8761597, -486.1722107, 808.1065674, -1296.5971680, 1299.0480957
3: -840.8578491, 717.4633789, -836.2380981, 713.1555786, -1554.0134277, 1553.7014160
4: -613.4078979, 824.6610107, -610.1495361, 819.8035278, -1433.2114258, 1434.8104248

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2042463, upper bound: 1339.2022582
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2051622, upper bound: 1339.2024900
time: 1.05 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -239.5506287, 216.3187408, -254.5586243, 230.2890320, -469.8396606, 470.8772888
1: -915.2257080, 780.8618774, -972.3587036, 830.7573242, -1745.9830322, 1753.2204590
2: -486.2151184, 809.0550537, -518.7409058, 860.8017578, -1347.0168457, 1327.7958984
3: -836.7950439, 714.0665283, -888.3858643, 759.3882446, -1596.1832275, 1602.4523926
4: -610.6746826, 820.8718872, -648.8429565, 873.5209351, -1484.1949463, 1469.7145996

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2098603, upper bound: 1339.2047596
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2099840, upper bound: 1339.2051603
time: 1.18 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -248.6404266, 224.6979828, -239.5235901, 216.2392426, -464.8796692, 464.2215576
1: -950.6085205, 810.5376587, -915.5968018, 780.3034668, -1730.9119873, 1726.1345215
2: -505.8209534, 840.9071655, -486.4881897, 808.6171875, -1314.4377441, 1327.3953857
3: -868.4695435, 741.5040283, -836.5622559, 713.6947632, -1582.1641846, 1578.0662842
4: -633.4951782, 853.2093506, -610.4876709, 820.3562012, -1453.8511963, 1463.6970215

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2042463, upper bound: 1339.2022582
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2050664, upper bound: 1339.2024887
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -247.6156769, 223.8017883, -254.7207031, 230.4812012, -478.0968628, 478.5224915
1: -946.3405762, 807.4158936, -972.9044189, 831.5322266, -1777.8728027, 1780.3203125
2: -503.7107849, 837.3676758, -519.1157837, 861.4462891, -1365.1571045, 1356.4832764
3: -864.8913574, 738.4001465, -888.8688965, 759.9976196, -1624.8887939, 1627.2686768
4: -631.1087646, 849.7878418, -649.2426147, 874.2158203, -1505.3244629, 1499.0301514

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2089633, upper bound: 1339.2047596
time: 1.10 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2090605, upper bound: 1339.2051421
time: 1.07 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -241.0880585, 217.6478119, -272.1485596, 245.7632751, -486.8512878, 489.7963867
1: -921.0563354, 785.5489502, -1041.4295654, 888.3779297, -1809.4340820, 1826.9785156
2: -489.3231812, 814.3442383, -551.9671021, 917.3292236, -1406.6522217, 1366.3112793
3: -842.3615723, 718.3698120, -950.6432495, 812.4470825, -1654.8085938, 1669.0129395
4: -614.6597900, 826.1160278, -693.7713013, 931.4503784, -1546.1099854, 1519.8870850

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2096769, upper bound: 1339.2010971
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2096983, upper bound: 1339.2010998
time: 1.40 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -242.0287170, 218.6124115, -271.5109863, 245.1866150, -487.2152405, 490.1233826
1: -924.5228271, 789.1265259, -1038.9592285, 886.4127197, -1810.9353027, 1828.0855713
2: -491.1266174, 817.7171631, -550.3352051, 914.7817993, -1405.9082031, 1368.0521240
3: -845.7655029, 721.5553589, -948.7453613, 810.4855957, -1656.2509766, 1670.3006592
4: -617.1549683, 829.7791138, -692.3018188, 929.1370850, -1546.2919922, 1522.0806885

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2106033, upper bound: 1339.2058964
time: 1.03 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2133936, upper bound: 1339.2098230
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -247.9262238, 223.8672028, -272.2484436, 245.8974304, -493.8236694, 496.1156311
1: -947.3966064, 808.2659912, -1041.7568359, 888.9150391, -1836.3116455, 1850.0228271
2: -503.2750549, 837.8613892, -552.2413940, 917.7733154, -1421.0483398, 1390.1022949
3: -866.0735474, 739.1180420, -950.9048462, 812.8472290, -1678.9206543, 1690.0227051
4: -631.9456177, 850.2402344, -694.0126343, 931.9260254, -1563.8715820, 1544.2524414

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2084659, upper bound: 1339.2010541
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2073289, upper bound: 1339.2009815
time: 1.13 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -249.3897858, 225.2522278, -271.3371277, 245.0803833, -494.4701538, 496.5893555
1: -952.9355469, 813.1968994, -1038.2214355, 886.1007690, -1839.0360107, 1851.4180908
2: -506.2121582, 842.8450317, -550.0541382, 914.3081055, -1420.5202637, 1392.8990479
3: -871.4060669, 743.6499023, -948.0367432, 810.0859375, -1681.4919434, 1691.6864014
4: -635.8555908, 855.3915405, -691.8372803, 928.6932373, -1564.5488281, 1547.2287598

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2087984, upper bound: 1339.2057300
time: 1.25 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2107886, upper bound: 1339.2096560
time: 1.21 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -259.4102783, 234.0572205, -240.3780670, 217.0428467, -476.4531250, 474.4352417
1: -992.0786133, 844.7562256, -918.6322632, 783.2313843, -1775.3100586, 1763.3883057
2: -527.0791016, 875.6035767, -488.0392761, 811.8165894, -1338.8957520, 1363.6427002
3: -906.3832397, 772.9333496, -839.6737061, 716.4165649, -1622.7998047, 1612.6069336
4: -660.8869019, 888.5584106, -612.7609253, 823.6026611, -1484.4895020, 1501.3193359

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2042380, upper bound: 1339.2022572
time: 0.90 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2052727, upper bound: 1339.2024900
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -257.7057495, 232.5542450, -255.4536133, 231.1582794, -488.8640137, 488.0078125
1: -985.2550049, 839.6151123, -975.5214844, 834.1405029, -1819.3955078, 1815.1364746
2: -523.5487671, 869.8751831, -520.3741455, 864.1233521, -1387.6716309, 1390.2492676
3: -900.3822632, 767.9545898, -891.5982056, 762.3419800, -1662.7239990, 1659.5527344
4: -656.7707520, 882.7957764, -651.3016968, 876.9276123, -1533.6983643, 1534.0974121

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2085055, upper bound: 1339.2047499
time: 0.91 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2098123, upper bound: 1339.2051603
time: 1.32 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -264.0706482, 238.7200928, -240.4463806, 217.1290894, -481.1997375, 479.1664429
1: -1009.8776855, 861.2351074, -918.7933350, 783.7332153, -1793.6108398, 1780.0284424
2: -537.6168213, 893.0028076, -488.2299805, 812.0334473, -1349.6499023, 1381.2327881
3: -922.5536499, 787.6129761, -839.8155518, 716.7209473, -1639.2745361, 1627.4284668
4: -672.8308105, 906.1829834, -612.9738770, 823.8774414, -1496.7082520, 1519.1567383

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2042289, upper bound: 1339.2022572
time: 1.09 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2050679, upper bound: 1339.2024887
time: 1.07 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -262.8496094, 237.6597748, -255.4538879, 231.1876526, -494.0372620, 493.1136475
1: -1004.9044189, 857.5356445, -975.4573364, 834.3588257, -1839.2631836, 1832.9929199
2: -535.1123047, 888.7391357, -520.3934326, 864.1655273, -1399.2775879, 1409.1325684
3: -918.3172607, 783.9658203, -891.5268555, 762.4269409, -1680.7441406, 1675.4924316
4: -669.9252319, 902.0311890, -651.2845459, 877.0018311, -1546.9270020, 1553.3156738

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2083261, upper bound: 1339.2047499
time: 1.04 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2089640, upper bound: 1339.2051421
time: 1.14 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -259.2516785, 233.8560486, -273.0415649, 246.5357208, -505.7873230, 506.8976135
1: -991.1159668, 844.2283936, -1044.5671387, 891.1723022, -1882.2883301, 1888.7955322
2: -526.0892334, 874.7315674, -553.8176270, 920.1645508, -1446.2536621, 1428.5490723
3: -906.0498657, 772.0703735, -953.8561401, 814.8137207, -1720.8635254, 1725.9262695
4: -660.8500977, 887.6252441, -696.1738281, 934.5176392, -1595.3675537, 1583.7990723

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2096632, upper bound: 1339.2010971
time: 0.98 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2097081, upper bound: 1339.2011416
time: 1.11 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -260.0804138, 234.7059479, -272.6542053, 246.3631439, -506.4435425, 507.3601379
1: -994.1173706, 847.5324707, -1042.9739990, 890.5830688, -1884.7004395, 1890.5063477
2: -527.6762695, 877.8481445, -552.8033447, 919.3504639, -1447.0267334, 1430.6514893
3: -908.9639282, 775.0811157, -952.7611694, 814.1863403, -1723.1502686, 1727.8422852
4: -662.9977417, 890.9519653, -695.3567505, 933.8190918, -1596.8168945, 1586.3087158

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2088892, upper bound: 1339.2056808
time: 1.04 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2129888, upper bound: 1339.2098385
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -263.2416077, 237.7662659, -273.0652771, 246.5904388, -509.8319702, 510.8315430
1: -1006.5051880, 858.5684204, -1044.6037598, 891.4320068, -1897.9368896, 1903.1721191
2: -534.6439209, 889.1583252, -553.9404907, 920.3229370, -1454.9667969, 1443.0988770
3: -919.8841553, 784.8515625, -953.8467407, 814.9497070, -1734.8338623, 1738.6982422
4: -671.0628662, 902.4487305, -696.2216187, 934.6914062, -1605.7542725, 1598.6704102

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2079198, upper bound: 1339.2010541
time: 1.16 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2072499, upper bound: 1339.2009942
time: 1.10 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -264.3241577, 238.8229523, -272.4212646, 246.1972046, -510.5213623, 511.2442017
1: -1010.3437500, 862.3834839, -1042.0189209, 890.0526123, -1900.3961182, 1904.4023438
2: -536.5979614, 893.0515137, -552.3860474, 918.6443481, -1455.2421875, 1445.4375000
3: -923.8197021, 788.3522339, -951.8493042, 813.5937500, -1737.4134521, 1740.2015381
4: -673.9197998, 906.4873657, -694.7325439, 933.1500244, -1607.0698242, 1601.2197266

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2003374, upper bound: 1339.2025467
time: 1.00 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2096560, upper bound: 1339.2096560
time: 0.93 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.08 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 3, lower bound: -1339.2042463, upper bound: 1339.2022582
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 3, lower bound: -1339.2051622, upper bound: 1339.2024900
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 3, lower bound: -1339.2098603, upper bound: 1339.2047596
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 3, lower bound: -1339.2099840, upper bound: 1339.2051603
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 3, lower bound: -1339.2042463, upper bound: 1339.2022582
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 3, lower bound: -1339.2050664, upper bound: 1339.2024887
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 3, lower bound: -1339.2089633, upper bound: 1339.2047596
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 3, lower bound: -1339.2090605, upper bound: 1339.2051421
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 3, lower bound: -1339.2096769, upper bound: 1339.2010971
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 3, lower bound: -1339.2096983, upper bound: 1339.2010998
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 3, lower bound: -1339.2106033, upper bound: 1339.2058964
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 3, lower bound: -1339.2133936, upper bound: 1339.2098230
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 3, lower bound: -1339.2084659, upper bound: 1339.2010541
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 3, lower bound: -1339.2073289, upper bound: 1339.2009815
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 3, lower bound: -1339.2087984, upper bound: 1339.2057300
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 3, lower bound: -1339.2107886, upper bound: 1339.2096560
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 3, lower bound: -1339.2042380, upper bound: 1339.2022572
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 3, lower bound: -1339.2052727, upper bound: 1339.2024900
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 3, lower bound: -1339.2085055, upper bound: 1339.2047499
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 3, lower bound: -1339.2098123, upper bound: 1339.2051603
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 3, lower bound: -1339.2042289, upper bound: 1339.2022572
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 3, lower bound: -1339.2050679, upper bound: 1339.2024887
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 3, lower bound: -1339.2083261, upper bound: 1339.2047499
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 3, lower bound: -1339.2089640, upper bound: 1339.2051421
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 3, lower bound: -1339.2096632, upper bound: 1339.2010971
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 3, lower bound: -1339.2097081, upper bound: 1339.2011416
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 3, lower bound: -1339.2088892, upper bound: 1339.2056808
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 3, lower bound: -1339.2129888, upper bound: 1339.2098385
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 3, lower bound: -1339.2079198, upper bound: 1339.2010541
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 3, lower bound: -1339.2072499, upper bound: 1339.2009942
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 3, lower bound: -1339.2003374, upper bound: 1339.2025467
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.08
Output dim: 3, lower bound: -1339.2096560, upper bound: 1339.2096560

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -237.3047638, 214.1831970, -235.9382172, 212.8277435, -450.1325073, 450.1213989
1: -907.2437744, 772.8765869, -902.1788330, 767.6538696, -1674.8977051, 1675.0551758
2: -481.9154358, 801.5621948, -479.2489624, 796.2142944, -1278.1297607, 1280.8111572
3: -828.7713013, 707.1593018, -824.0557251, 702.3327637, -1531.1036377, 1531.2150879
4: -604.5304565, 812.9217529, -601.2230225, 807.4223633, -1411.9528809, 1414.1447754

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2042463, upper bound: 1339.2022582
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2042463, upper bound: 1339.2022582
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -235.9300690, 212.8904419, -236.5107727, 213.4085541, -449.3385315, 449.4011841
1: -901.9426880, 768.4196777, -904.2285156, 769.9730225, -1671.9157715, 1672.6481934
2: -478.6282349, 796.4040527, -480.1557617, 798.1550293, -1276.7832031, 1276.5598145
3: -824.2557983, 702.9916992, -826.1903687, 704.3865356, -1528.6420898, 1529.1820068
4: -601.2334595, 807.9400635, -602.7881470, 809.6685791, -1410.9016113, 1410.7281494

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.2023548, upper bound: 1339.1989817
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2051610, upper bound: 1339.2024900
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -236.2012329, 213.2093811, -251.2098541, 227.1486053, -463.3498535, 464.4192200
1: -902.6368408, 769.5385742, -959.6779785, 819.2429199, -1721.8795166, 1729.2164307
2: -479.6101685, 797.8804321, -512.0927734, 849.4507446, -1329.0606689, 1309.9730225
3: -824.8554077, 703.8723755, -876.5418091, 748.8633423, -1573.7187500, 1580.4138184
4: -601.9377441, 809.2599487, -640.2004395, 861.6387329, -1463.5764160, 1449.4604492

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2097012, upper bound: 1339.2038879
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2097768, upper bound: 1339.2047175
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -234.8213501, 211.9340210, -251.6724854, 227.6168365, -462.4381714, 463.6065063
1: -897.2947998, 765.1793823, -961.4297485, 821.1757812, -1718.4705811, 1726.6088867
2: -476.4437866, 792.7937622, -512.7238159, 850.8583984, -1327.3021240, 1305.5170898
3: -820.3529053, 699.7548218, -878.3684692, 750.6483154, -1571.0012207, 1578.1230469
4: -598.6237793, 804.3310547, -641.4797974, 863.3958130, -1462.0192871, 1445.8105469

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2031018, upper bound: 1339.1989479
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2099840, upper bound: 1339.2051568
time: 1.18 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -244.6216125, 220.9993591, -236.1850128, 213.0997772, -457.7213745, 457.1843262
1: -935.3531494, 797.1328125, -903.0286865, 768.7799072, -1704.1330566, 1700.1614990
2: -498.0315247, 827.8903809, -479.8143005, 797.1250610, -1295.1564941, 1307.7047119
3: -854.0465088, 729.3665771, -824.8171387, 703.2443848, -1557.2908936, 1554.1837158
4: -622.9319458, 839.7274780, -601.8699341, 808.4053345, -1431.3372803, 1441.5971680

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1965277, upper bound: 1339.1976641
time: 1.21 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2042463, upper bound: 1339.2022582
time: 1.06 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -244.1410522, 220.5067444, -236.5099182, 213.4487152, -457.5897827, 457.0166016
1: -933.5639648, 795.4483032, -904.1198120, 770.2955322, -1703.8594971, 1699.5681152
2: -496.5055847, 825.2699585, -480.2152405, 798.2130127, -1294.7183838, 1305.4852295
3: -852.8737183, 727.7341919, -826.0791016, 704.5501099, -1557.4237061, 1553.8132324
4: -622.0622559, 837.3418579, -602.8086548, 809.7894287, -1431.8516846, 1440.1502686

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.2022980, upper bound: 1339.1992725
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2050658, upper bound: 1339.2024887
time: 1.26 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -243.6106415, 220.1210480, -251.4703522, 227.4273834, -471.0380249, 471.5914001
1: -931.1694336, 794.0617065, -960.6253662, 820.3049927, -1751.4741211, 1754.6870117
2: -495.9736938, 824.3673096, -512.6611328, 850.3966064, -1346.3702393, 1337.0284424
3: -850.5075684, 726.3127441, -877.3654175, 749.7482300, -1600.2558594, 1603.6781006
4: -620.6094971, 836.2466431, -640.8462524, 862.6382446, -1483.2475586, 1477.0927734

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1969243, upper bound: 1339.1970531
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2089633, upper bound: 1339.2047512
time: 1.16 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -243.1540375, 219.6455231, -251.7054749, 227.6916656, -470.8457031, 471.3510132
1: -929.4296875, 792.4547729, -961.4818726, 821.5283203, -1750.9580078, 1753.9365234
2: -494.4725342, 821.9014893, -512.8363037, 851.0585327, -1345.5310059, 1334.7375488
3: -849.4176636, 724.7445068, -878.3920898, 750.8691406, -1600.2868652, 1603.1365967
4: -619.7689819, 834.0583496, -641.5484619, 863.6408691, -1483.4099121, 1475.6066895

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2025521, upper bound: 1339.1989479
time: 1.17 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2090605, upper bound: 1339.2051421
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -239.3489838, 216.0962677, -271.0399475, 244.7756042, -484.1245728, 487.1362000
1: -914.4100342, 779.9654541, -1037.1865234, 884.8594360, -1799.2695312, 1817.1519775
2: -485.8578491, 808.5031738, -549.7502441, 913.5739746, -1399.4318848, 1358.2530518
3: -836.2752075, 713.2137451, -946.7562256, 809.1876831, -1645.4628906, 1659.9696045
4: -610.2350464, 820.1897583, -690.9441528, 927.6565552, -1537.8916016, 1511.1339111

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2081828, upper bound: 1339.2009873
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2088187, upper bound: 1339.2009872
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -241.3318329, 217.8314819, -269.8409729, 243.6572113, -484.9890442, 487.6724548
1: -922.2141113, 786.0805664, -1032.6168213, 880.7841797, -1802.9981689, 1818.6973877
2: -489.9196472, 815.2639771, -547.2683105, 909.4194336, -1399.3389893, 1362.5322266
3: -843.2152710, 719.1693726, -942.6015015, 805.5321655, -1648.7474365, 1661.7708740
4: -615.1528931, 827.0989990, -687.8822632, 923.4545898, -1538.6074219, 1514.9810791

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2080872, upper bound: 1339.2009361
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2088158, upper bound: 1339.2009361
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -239.5767822, 216.4208374, -270.3170471, 244.1212158, -483.6979065, 486.7378845
1: -915.0674438, 781.1777344, -1034.3814697, 882.5582275, -1797.6257324, 1815.5592041
2: -486.1841125, 809.6193848, -547.8984375, 910.8295898, -1397.0133057, 1357.5177002
3: -837.1600342, 714.3130493, -944.5650024, 806.9660034, -1644.1259766, 1658.8780518
4: -610.9019775, 821.5448608, -689.2398682, 925.1299438, -1536.0316162, 1510.7846680

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.2016633, upper bound: 1339.1992737
time: 1.18 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2067673, upper bound: 1339.2032120
time: 1.00 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2034158, upper bound: 1339.1991252
time: 1.03 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -245.9664307, 221.8299255, -269.5517578, 243.4051666, -489.3715820, 491.3816223
1: -940.4005127, 801.0568237, -1031.4243164, 879.9791260, -1820.3793945, 1832.4812012
2: -498.6010132, 829.4559326, -546.2996216, 908.2190552, -1406.8198242, 1375.7556152
3: -859.8373413, 732.7447510, -941.9313965, 804.6390381, -1664.4763184, 1674.6761475
4: -627.1580811, 841.7766113, -687.3018188, 922.4589233, -1549.6169434, 1529.0783691

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2130334, upper bound: 1339.2097900
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2130548, upper bound: 1339.2092394
time: 1.05 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -246.2054138, 222.3442078, -271.1900024, 244.9567871, -491.1622009, 493.5342102
1: -940.8417969, 802.7692871, -1037.7127686, 885.5646362, -1826.4063721, 1840.4818115
2: -499.8439331, 832.1165161, -550.1250000, 914.1945801, -1414.0384521, 1382.2414551
3: -860.0316772, 734.0585327, -947.1942749, 809.7444458, -1669.7761230, 1681.2524414
4: -627.5264282, 844.4221191, -691.3105469, 928.3106079, -1555.8370361, 1535.7326660

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2081427, upper bound: 1339.2009868
time: 1.15 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2083560, upper bound: 1339.2009829
time: 0.94 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -246.8937531, 222.9100800, -269.8786316, 243.7348328, -490.6286011, 492.7886963
1: -943.5288086, 804.7462158, -1032.6997070, 881.1210327, -1824.6497803, 1837.4459229
2: -501.1356812, 834.2876587, -547.4160156, 909.6539917, -1410.7896729, 1381.7036133
3: -862.4740601, 736.0274048, -942.6434326, 805.7471313, -1668.2210693, 1678.6707764
4: -629.2427979, 846.6210327, -687.9649658, 923.7192383, -1552.9620361, 1534.5859375

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2071584, upper bound: 1339.2008633
time: 0.94 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2071933, upper bound: 1339.2008633
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -247.1167450, 223.2029266, -269.9780579, 243.8639374, -490.9806519, 493.1809692
1: -944.1835327, 805.7330933, -1032.9934082, 881.7029419, -1825.8864746, 1838.7265625
2: -501.6134644, 835.2603760, -547.3057251, 909.8113403, -1411.4248047, 1382.5656738
3: -863.4346313, 736.8588257, -943.2742310, 806.0728149, -1669.5072021, 1680.1326904
4: -630.0420532, 847.6860962, -688.3608398, 924.1375732, -1554.1795654, 1536.0468750

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1938465, upper bound: 1339.1938122
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1938462, upper bound: 1339.1934115
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -254.4377441, 229.6097260, -269.4972839, 243.4037170, -497.8413391, 499.1069946
1: -972.9774170, 828.7832642, -1031.1522217, 880.0431519, -1853.0205078, 1859.9354248
2: -516.5747681, 858.8543701, -546.2585449, 908.1178589, -1424.6925049, 1405.1129150
3: -889.3673096, 758.1272583, -941.6437988, 804.5829468, -1693.9501953, 1699.7709961
4: -648.7449951, 871.7431030, -687.1441650, 922.4017334, -1571.1467285, 1558.8872070

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2093359, upper bound: 1339.2096048
time: 1.24 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2093540, upper bound: 1339.2090508
time: 1.14 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -253.3698883, 228.3346252, -236.4245911, 213.3100739, -466.6799622, 464.7591858
1: -969.0567627, 823.7134399, -903.6866455, 769.5877686, -1738.6445312, 1727.4000244
2: -514.7943115, 854.2956543, -480.1181641, 798.0116577, -1312.8059082, 1334.4136963
3: -885.2359009, 753.5993042, -825.7647095, 703.9492798, -1589.1851807, 1579.3640137
4: -645.4916382, 866.5679321, -602.5792236, 809.4914551, -1454.9829102, 1469.1469727

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2042380, upper bound: 1339.2022572
time: 1.15 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2042380, upper bound: 1339.2022572
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -254.6579132, 229.6741943, -237.6986084, 214.5821228, -469.2398987, 467.3727722
1: -974.0232544, 829.0515747, -908.4598999, 774.3543091, -1748.3771973, 1737.5114746
2: -517.2035522, 859.3847656, -482.5328674, 802.6871948, -1319.8906250, 1341.9174805
3: -889.8092651, 758.6160278, -830.3549194, 708.3197632, -1598.1290283, 1588.9709473
4: -648.7110596, 872.0397339, -605.9510498, 814.3043213, -1463.0150146, 1477.9906006

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2024763, upper bound: 1339.1993985
time: 1.08 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2052714, upper bound: 1339.2024900
time: 1.02 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -251.8082428, 226.9605255, -251.5904388, 227.5140228, -479.3222046, 478.5509338
1: -962.8203125, 819.0346680, -960.8529663, 820.8314209, -1783.6517334, 1779.8874512
2: -511.5867920, 848.9863892, -512.6615601, 850.6417236, -1362.2283936, 1361.6479492
3: -879.7473145, 749.0356445, -877.9764404, 750.1948242, -1629.9421387, 1627.0120850
4: -641.7359009, 861.3130493, -641.3248291, 863.1361694, -1504.8720703, 1502.6379395

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2083990, upper bound: 1339.2038829
time: 1.16 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2084982, upper bound: 1339.2047031
time: 1.05 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -252.9685211, 228.1843414, -252.7706146, 228.6759949, -481.6445312, 480.9549561
1: -967.2585449, 823.9429321, -965.2887573, 825.1876221, -1792.4461670, 1789.2316895
2: -513.6930542, 853.7008667, -514.8272705, 854.9105835, -1368.6036377, 1368.5280762
3: -883.8768921, 753.6647949, -882.3106689, 754.1736450, -1638.0505371, 1635.9754639
4: -644.6378784, 866.3228149, -644.4810791, 867.5439453, -1512.1817627, 1510.8037109

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2029659, upper bound: 1339.1990003
time: 1.13 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2098123, upper bound: 1339.2051568
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -258.6899719, 233.6429138, -236.6425476, 213.5314636, -472.2214050, 470.2854614
1: -989.4574585, 842.6859131, -904.4378662, 770.5770874, -1760.0345459, 1747.1236572
2: -526.9299927, 874.6837769, -480.5927734, 798.7371216, -1325.6669922, 1355.2764893
3: -903.6679077, 770.6726685, -826.4243164, 704.7115479, -1608.3793945, 1597.0968018
4: -658.9655151, 887.1510010, -603.1655884, 810.2830200, -1469.2485352, 1490.3164062

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2042289, upper bound: 1339.2022572
time: 1.02 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2042289, upper bound: 1339.2022572
time: 1.18 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -259.4906006, 234.4779358, -237.6953583, 214.6058502, -474.0964050, 472.1732483
1: -992.4190674, 845.8963623, -908.3462524, 774.6301270, -1767.0490723, 1754.2426758
2: -528.1212769, 877.2385254, -482.5773010, 802.6571655, -1330.7783203, 1359.8154297
3: -906.6972656, 773.6320801, -830.2467041, 708.4182129, -1615.1154785, 1603.8786621
4: -661.1355591, 890.1708984, -605.9622803, 814.3333740, -1475.4689941, 1496.1330566

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.2023279, upper bound: 1339.1994022
time: 1.38 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2050677, upper bound: 1339.2024887
time: 1.14 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -257.5476379, 232.6576996, -251.7248383, 227.6700745, -485.2176819, 484.3825378
1: -984.8438110, 839.2739868, -961.3205566, 821.4846191, -1806.3281250, 1800.5944824
2: -524.5983887, 870.6354370, -512.9650269, 851.1552734, -1375.7536621, 1383.5999756
3: -899.6264648, 767.2828979, -878.3693848, 750.6828613, -1650.3093262, 1645.6523438
4: -656.2755737, 883.2409058, -641.6647949, 863.6875610, -1519.9630127, 1524.9055176

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2080766, upper bound: 1339.2038699
time: 0.95 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2083231, upper bound: 1339.2047031
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -258.2918091, 233.4325409, -252.6890564, 228.6412201, -486.9329834, 486.1215820
1: -987.5647583, 842.2702026, -964.9472046, 825.1721802, -1812.7369385, 1807.2174072
2: -525.6531982, 873.0382690, -514.6800537, 854.7035522, -1380.3565674, 1387.7178955
3: -902.5319824, 770.0437622, -881.9588013, 754.0435181, -1656.5754395, 1652.0025635
4: -658.2893677, 886.0858765, -644.2730713, 867.3659668, -1525.6552734, 1530.3588867

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2025521, upper bound: 1339.1990003
time: 0.94 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2089640, upper bound: 1339.2051421
time: 0.94 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -257.4177246, 232.2189331, -271.8168335, 245.4395447, -502.8572693, 504.0357361
1: -984.1553345, 838.3179932, -1039.8815918, 887.2280884, -1871.3834229, 1878.1993408
2: -522.3862305, 868.5393066, -551.3753052, 916.0249634, -1438.4110107, 1419.9145508
3: -899.6546021, 766.6091309, -949.5623169, 811.1728516, -1710.8273926, 1716.1711426
4: -656.1762085, 881.3200684, -693.0509644, 930.3133545, -1586.4892578, 1574.3710938

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2081828, upper bound: 1339.2010073
time: 1.02 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2088187, upper bound: 1339.2010073
time: 0.94 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -261.3445435, 235.7145386, -270.9028015, 244.5852966, -505.9298401, 506.6173401
1: -999.4755249, 850.8589478, -1036.3908691, 884.1458130, -1883.6213379, 1887.2497559
2: -530.4397583, 882.0534058, -549.4633789, 912.8460693, -1443.2857666, 1431.5168457
3: -913.3894043, 778.5126343, -946.4036865, 808.4071655, -1721.7966309, 1724.9160156
4: -665.9887085, 894.8601685, -690.7125854, 927.1137695, -1593.1024170, 1585.5727539

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2080872, upper bound: 1339.2010002
time: 1.10 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2088158, upper bound: 1339.2010002
time: 1.08 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -258.1245117, 232.9502563, -271.4865417, 245.3209534, -503.4454651, 504.4367981
1: -986.6390381, 841.1795654, -1038.5008545, 886.8162231, -1873.4553223, 1879.6804199
2: -523.6677246, 871.3186035, -550.4214478, 915.4736938, -1439.1412354, 1421.7399902
3: -902.1073608, 769.2978516, -948.6759033, 810.7420654, -1712.8493652, 1717.9736328
4: -657.9575806, 884.3269043, -692.3682861, 929.8890381, -1587.8466797, 1576.6949463

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2062814, upper bound: 1339.2031027
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2034183, upper bound: 1339.1991154
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -263.9983826, 237.9300842, -270.6961365, 244.6007843, -508.5990601, 508.6262207
1: -1009.9464722, 859.5163574, -1035.4627686, 884.1697388, -1894.1160889, 1894.9791260
2: -535.2799683, 889.6398926, -548.8090820, 912.8270874, -1448.1070557, 1438.4488525
3: -922.9818115, 786.3053589, -946.0038452, 808.3579712, -1731.3393555, 1732.3088379
4: -672.9878540, 902.9624634, -690.4107056, 927.1850586, -1600.1727295, 1593.3730469

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2129358, upper bound: 1339.2098121
time: 1.21 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2129529, upper bound: 1339.2092406
time: 0.98 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -261.3801270, 236.0993500, -271.9754333, 245.6219025, -507.0020142, 508.0747681
1: -999.4566040, 852.5791626, -1040.4434814, 887.9504395, -1887.4069824, 1893.0227051
2: -530.9442139, 882.8313599, -551.7736816, 916.6600952, -1447.6042480, 1434.6046143
3: -913.4161377, 779.3284302, -950.0261230, 811.7322998, -1725.1484375, 1729.3544922
4: -666.2922363, 896.0417480, -693.4400635, 930.9725952, -1597.2646484, 1589.4818115

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2078933, upper bound: 1339.2009905
time: 1.08 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2078876, upper bound: 1339.2009829
time: 0.97 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -263.9794006, 238.3972931, -270.8135071, 244.5363464, -508.5157471, 509.2108154
1: -1009.5444336, 860.7710571, -1035.9877930, 884.0319214, -1893.5764160, 1896.7587891
2: -536.2728271, 891.5035400, -549.3558350, 912.6168213, -1448.8892822, 1440.8593750
3: -922.5599976, 787.1066284, -945.9981079, 808.2012939, -1730.7612305, 1733.1047363
4: -672.8263550, 904.8679199, -690.4738770, 926.8946533, -1599.7209473, 1595.3416748

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2069758, upper bound: 1339.2008808
time: 0.99 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2069767, upper bound: 1339.2008808
time: 0.93 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -262.1231995, 236.8440552, -271.0791321, 244.9949951, -507.1181946, 507.9231262
1: -1001.8856201, 855.2045898, -1036.8602295, 885.7108154, -1887.5964355, 1892.0646973
2: -532.1710205, 885.7313232, -549.6719360, 914.2020874, -1446.3726807, 1435.4033203
3: -916.0914917, 781.8232422, -947.1485596, 809.6256104, -1725.7170410, 1728.9718018
4: -668.2664795, 899.0598145, -691.3054810, 928.6404419, -1596.9069824, 1590.3652344

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1786270, upper bound: 1339.1831347
time: 0.92 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1786284, upper bound: 1339.1832511
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -269.4082336, 243.2724304, -270.5602722, 244.5213776, -513.9296265, 513.8327026
1: -1030.5457764, 878.1102905, -1034.8796387, 883.9518433, -1914.4973145, 1912.9897461
2: -547.4459229, 909.2052002, -548.5783691, 912.4403076, -1459.8857422, 1457.7835693
3: -941.9506226, 802.8761597, -945.4209595, 808.0493164, -1750.0000000, 1748.2968750
4: -686.9041748, 923.0275879, -690.0359497, 926.8364868, -1613.7407227, 1613.0634766

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2090572, upper bound: 1339.2096048
time: 1.10 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2090508, upper bound: 1339.2090508
time: 1.34 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.96 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2042463, upper bound: 1339.2022582
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2042463, upper bound: 1339.2022582
NS_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2023548, upper bound: 1339.1989817
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2051610, upper bound: 1339.2024900
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2097012, upper bound: 1339.2038879
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2097768, upper bound: 1339.2047175
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2031018, upper bound: 1339.1989479
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2099840, upper bound: 1339.2051568
NS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.1965277, upper bound: 1339.1976641
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2042463, upper bound: 1339.2022582
NS_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2022980, upper bound: 1339.1992725
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2050658, upper bound: 1339.2024887
NS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.1969243, upper bound: 1339.1970531
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2089633, upper bound: 1339.2047512
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2025521, upper bound: 1339.1989479
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2090605, upper bound: 1339.2051421
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2081828, upper bound: 1339.2009873
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2088187, upper bound: 1339.2009872
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2080872, upper bound: 1339.2009361
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2088158, upper bound: 1339.2009361
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2067673, upper bound: 1339.2032120
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2034158, upper bound: 1339.1991252
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2130334, upper bound: 1339.2097900
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2130548, upper bound: 1339.2092394
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2081427, upper bound: 1339.2009868
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2083560, upper bound: 1339.2009829
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2071584, upper bound: 1339.2008633
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2071933, upper bound: 1339.2008633
NS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.1938465, upper bound: 1339.1938122
NS_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.1938462, upper bound: 1339.1934115
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2093359, upper bound: 1339.2096048
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2093540, upper bound: 1339.2090508
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2042380, upper bound: 1339.2022572
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2042380, upper bound: 1339.2022572
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2024763, upper bound: 1339.1993985
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2052714, upper bound: 1339.2024900
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2083990, upper bound: 1339.2038829
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2084982, upper bound: 1339.2047031
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2029659, upper bound: 1339.1990003
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2098123, upper bound: 1339.2051568
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2042289, upper bound: 1339.2022572
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2042289, upper bound: 1339.2022572
NS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2023279, upper bound: 1339.1994022
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2050677, upper bound: 1339.2024887
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2080766, upper bound: 1339.2038699
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2083231, upper bound: 1339.2047031
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2025521, upper bound: 1339.1990003
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2089640, upper bound: 1339.2051421
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2081828, upper bound: 1339.2010073
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2088187, upper bound: 1339.2010073
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2080872, upper bound: 1339.2010002
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2088158, upper bound: 1339.2010002
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2062814, upper bound: 1339.2031027
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2034183, upper bound: 1339.1991154
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2129358, upper bound: 1339.2098121
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2129529, upper bound: 1339.2092406
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2078933, upper bound: 1339.2009905
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2078876, upper bound: 1339.2009829
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2069758, upper bound: 1339.2008808
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2069767, upper bound: 1339.2008808
NS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.1786270, upper bound: 1339.1831347
NS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.1786284, upper bound: 1339.1832511
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2090572, upper bound: 1339.2096048
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -1339.2090508, upper bound: 1339.2090508

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -237.3047638, 214.1831970, -233.7981873, 210.8998566, -448.2046204, 447.9813843
1: -907.2437744, 772.8765869, -894.1912842, 760.6896362, -1667.9333496, 1667.0677490
2: -481.9154358, 801.5621948, -474.8912659, 788.8543091, -1270.7696533, 1276.4534912
3: -828.7713013, 707.1593018, -816.6022949, 696.0002441, -1524.7714844, 1523.7615967
4: -604.5304565, 812.9217529, -595.6613770, 799.9719849, -1404.5021973, 1408.5831299

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2042463, upper bound: 1339.2022582
time: 1.02 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2042463, upper bound: 1339.2022582
time: 1.08 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -237.3047638, 214.1831970, -252.9692535, 228.1091156, -465.4138794, 467.1524658
1: -907.2437744, 772.8765869, -967.9032593, 822.6008911, -1729.8447266, 1740.7797852
2: -481.9154358, 801.5621948, -514.4145508, 853.3258057, -1335.2412109, 1315.9768066
3: -828.7713013, 707.1593018, -883.7941284, 752.7252808, -1581.4962158, 1590.9533691
4: -604.5304565, 812.9217529, -644.3358765, 865.5685425, -1470.0989990, 1457.2574463

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2042463, upper bound: 1339.2022582
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2042463, upper bound: 1339.2022582
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -234.1967163, 211.3095703, -241.6353302, 218.0816956, -452.2784119, 452.9448853
1: -895.3092041, 762.7138062, -924.6104126, 786.1004028, -1681.4096680, 1687.3240967
2: -475.0674744, 790.5087280, -491.1406250, 815.7705688, -1290.8376465, 1281.6491699
3: -818.2381592, 697.8071289, -844.3289185, 719.1422729, -1537.3801270, 1542.1359863
4: -596.8206787, 801.9503784, -615.7349243, 827.2645264, -1424.0852051, 1417.6851807

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.2014705, upper bound: 1339.1978982
time: 5.12 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.2016841, upper bound: 1339.1979171
time: 0.94 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -233.5647278, 210.8715210, -247.6743164, 223.9729462, -457.5375977, 458.5458374
1: -892.5504150, 761.0787354, -946.1846313, 807.7471313, -1700.2972412, 1707.2634277
2: -474.3175659, 789.0820923, -505.0440979, 837.6181030, -1311.9356689, 1294.1260986
3: -815.6904297, 696.0490723, -864.2315674, 738.2561035, -1553.9465332, 1560.2806396
4: -595.2653198, 800.3306274, -631.1803589, 849.6427002, -1444.9079590, 1431.5108643

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2094571, upper bound: 1339.2038879
time: 1.15 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2094571, upper bound: 1339.2038879
time: 0.97 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -234.5313873, 211.7286072, -252.3708344, 228.2576141, -462.7890015, 464.0994263
1: -896.2449951, 764.1987915, -964.3143921, 823.6394043, -1719.8843994, 1728.5130615
2: -476.2510681, 792.2809448, -514.4348145, 853.8645020, -1330.1156006, 1306.7158203
3: -819.0302734, 698.9386597, -880.6461182, 752.8311768, -1571.8610840, 1579.5844727
4: -597.7030640, 803.5903320, -643.0491943, 866.0272217, -1463.7302246, 1446.6392822

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2094648, upper bound: 1339.2047175
time: 1.37 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2094648, upper bound: 1339.2047175
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -233.1645660, 210.4548492, -249.6818848, 225.8227234, -458.9873047, 460.1366272
1: -890.8935547, 759.8012695, -953.7653198, 814.6942749, -1705.5877686, 1713.5666504
2: -473.1061707, 787.3293457, -508.7099915, 844.2980957, -1317.4041748, 1296.0389404
3: -814.5260010, 694.8657227, -871.3827515, 744.7558594, -1559.2817383, 1566.2482910
4: -594.4045410, 798.7790527, -636.3742676, 856.7180786, -1451.1225586, 1435.1533203

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1829972, upper bound: 1339.1783952
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1832652, upper bound: 1339.1784105
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -233.2628021, 210.5044556, -257.7619934, 233.4402618, -466.7030640, 468.2664185
1: -891.3226929, 760.0374146, -985.1275635, 841.7631226, -1733.0858154, 1745.1649170
2: -473.2203979, 787.4750366, -525.7597656, 872.8706665, -1346.0910645, 1313.2348633
3: -814.9359131, 695.0864868, -899.6400757, 769.0985107, -1584.0343018, 1594.7264404
4: -594.6630249, 798.9342651, -656.9469604, 885.5507812, -1480.2138672, 1455.8811035

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2086634, upper bound: 1339.2016510
time: 6.40 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2086551, upper bound: 1339.2016500
time: 1.14 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -242.7718201, 219.3253937, -241.7101746, 218.2052612, -460.9770813, 461.0355835
1: -928.2656250, 791.0827637, -924.8873901, 786.5913086, -1714.8565674, 1715.9702148
2: -494.2160034, 821.5944214, -491.5572510, 816.3314819, -1310.5474854, 1313.1516113
3: -847.6124878, 723.8580933, -844.3236694, 719.2167969, -1566.8292236, 1568.1817627
4: -618.2447510, 833.3337402, -615.8583984, 827.6066895, -1445.8514404, 1449.1920166

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1999745, upper bound: 1339.1972308
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1881098, upper bound: 1339.1935665
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1860509, upper bound: 1339.1905280
time: 1.09 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.2023064, upper bound: 1339.2001072
time: 0.98 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -242.3665161, 218.9036713, -241.7605743, 218.2443848, -460.6109009, 460.6642151
1: -926.7500000, 789.6498413, -924.9904175, 786.8001099, -1713.5500488, 1714.6402588
2: -492.8354492, 819.2442627, -491.4474182, 816.2046509, -1309.0400391, 1310.6916504
3: -846.7022705, 722.4666748, -844.6577148, 719.6785889, -1566.3806152, 1567.1243896
4: -617.5658569, 831.2762451, -616.0748901, 827.7726440, -1445.3385010, 1447.3510742

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.2015141, upper bound: 1339.1979115
time: 1.52 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.2016695, upper bound: 1339.1979143
time: 1.19 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -241.9462585, 218.6136627, -257.7338257, 233.4513245, -475.3975525, 476.3474731
1: -924.7830200, 788.6207275, -984.9910889, 841.7597656, -1766.5426025, 1773.6116943
2: -492.5121155, 818.6886597, -526.0054932, 873.0886841, -1365.6008301, 1344.6940918
3: -844.7272339, 721.3613892, -899.2246704, 768.9946899, -1613.7219238, 1620.5859375
4: -616.3955078, 830.5289917, -656.7557373, 885.5690918, -1501.9645996, 1487.2846680

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2077452, upper bound: 1339.2014797
time: 1.15 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1912712, upper bound: 1339.1954288
time: 1.15 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1983843, upper bound: 1339.1995214
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2062596, upper bound: 1339.2023680
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -241.6275787, 218.2694397, -249.6656647, 225.8535919, -467.4811096, 467.9349976
1: -923.5513916, 787.4309692, -953.6289673, 814.8833618, -1738.4344482, 1741.0599365
2: -491.3917542, 816.7991943, -508.7251587, 844.3275757, -1335.7193604, 1325.5244141
3: -844.0631104, 720.1804199, -871.2341919, 744.8281860, -1588.8909912, 1591.4145508
4: -615.8598633, 828.8786621, -636.3159180, 856.7901611, -1472.6499023, 1465.1945801

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1830141, upper bound: 1339.1784087
time: 1.25 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1830141, upper bound: 1339.1784105
time: 1.29 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -241.4635468, 218.1124725, -257.7294312, 233.4487152, -474.9122620, 475.8418884
1: -922.9350586, 786.9233398, -984.9179688, 841.8587646, -1764.7938232, 1771.8411865
2: -490.9530029, 816.1876221, -525.7435303, 872.8175049, -1363.7705078, 1341.9307861
3: -843.5426025, 719.7164307, -899.4412842, 769.1005249, -1612.6430664, 1619.1574707
4: -615.4883423, 828.2564087, -656.8605957, 885.5531616, -1501.0415039, 1485.1165771

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2078610, upper bound: 1339.2015836
time: 1.25 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2075002, upper bound: 1339.2015714
time: 1.06 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -235.0272675, 212.3256989, -262.2296143, 237.0839386, -472.1112061, 474.5552673
1: -897.6782837, 766.1102905, -1002.7100220, 856.5427246, -1754.2209473, 1768.8201904
2: -477.1947937, 794.4055786, -532.6450806, 886.1159058, -1363.3105469, 1327.0504150
3: -821.0321045, 700.5895996, -915.5655518, 783.2963257, -1604.3283691, 1616.1551514
4: -599.2575684, 805.9805908, -668.5641479, 899.5009766, -1498.7585449, 1474.5446777

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2081533, upper bound: 1339.2006775
time: 0.94 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2081533, upper bound: 1339.2009873
time: 1.11 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -236.6578827, 213.6956940, -266.5766602, 240.7951050, -477.4529114, 480.2723083
1: -904.1071167, 771.2742920, -1020.1044312, 870.4888916, -1774.5959473, 1791.3786621
2: -480.4222717, 799.5614624, -540.7216187, 898.8584595, -1379.2806396, 1340.2830811
3: -826.8369751, 705.2832642, -931.1166992, 796.0867310, -1622.9234619, 1636.3999023
4: -603.3530273, 811.1414185, -679.5273438, 912.7588501, -1516.1118164, 1490.6687012

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2068942, upper bound: 1339.2007119
time: 1.20 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2087322, upper bound: 1339.2009872
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -237.0065308, 214.0529633, -261.0028992, 235.9277649, -472.9342957, 475.0558167
1: -905.4521484, 772.2599487, -998.0052490, 852.3404541, -1757.7922363, 1770.2651367
2: -481.2441101, 801.1387329, -530.1058350, 881.8137817, -1363.0574951, 1331.2446289
3: -827.9541626, 706.5431519, -911.3208008, 779.5278320, -1607.4819336, 1617.8640137
4: -604.1469727, 812.8411255, -665.4411011, 895.1525879, -1499.2995605, 1478.2822266

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2061142, upper bound: 1339.2002327
time: 1.05 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2061142, upper bound: 1339.2009361
time: 1.01 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -238.6851959, 215.4649200, -265.4224243, 239.7167206, -478.4019165, 480.8872986
1: -912.0720215, 777.4750366, -1015.6848145, 866.5529175, -1778.6248779, 1793.1599121
2: -484.5766602, 806.4796143, -538.3458862, 894.8737793, -1379.4504395, 1344.8251953
3: -833.9307861, 711.3308716, -927.1198120, 792.5496216, -1626.4803467, 1638.4506836
4: -608.3818970, 818.2039185, -676.5797119, 908.7207642, -1517.1025391, 1494.7836914

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2051892, upper bound: 1339.2000242
time: 1.17 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2085259, upper bound: 1339.2008351
time: 1.42 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -236.8290253, 214.0003662, -266.0654297, 240.3565826, -477.1856079, 480.0657654
1: -904.5003052, 772.4071045, -1018.0449829, 868.9071655, -1773.4073486, 1790.4521484
2: -480.6513367, 800.5588379, -539.3090820, 896.7630005, -1377.4143066, 1339.8679199
3: -827.5521240, 706.2690430, -929.7396240, 794.4634399, -1622.0151367, 1636.0085449
4: -603.8888550, 812.3683472, -678.3757324, 910.8710938, -1514.7596436, 1490.7441406

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1970283, upper bound: 1339.1977394
time: 1.01 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2033213, upper bound: 1339.2007092
time: 1.03 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -236.8251190, 213.9705811, -272.7254028, 246.2179108, -483.0429688, 486.6959839
1: -904.5326538, 772.2912598, -1044.0148926, 889.7291870, -1794.2617188, 1816.3061523
2: -480.6272278, 800.4854736, -553.1292725, 919.2645874, -1399.8914795, 1353.6147461
3: -827.5596924, 706.1173706, -953.0559082, 813.7044067, -1641.2641602, 1659.1732178
4: -603.9217529, 812.2573242, -695.3570557, 933.2905273, -1537.2122803, 1507.6141357

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1984342, upper bound: 1339.1956135
time: 1.05 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2034158, upper bound: 1339.1991252
time: 1.14 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -244.6846008, 220.6765289, -267.5802307, 241.6364746, -486.3210754, 488.2567749
1: -935.4969482, 796.8967896, -1023.9000854, 873.6224976, -1809.1193848, 1820.7965088
2: -496.0357666, 825.1090698, -542.3331299, 901.4782715, -1397.5140381, 1367.4418945
3: -855.3458862, 728.9097290, -935.0212402, 798.7763672, -1654.1221924, 1663.9309082
4: -623.9010010, 837.3704224, -682.2860107, 915.6552734, -1539.5562744, 1519.6563721

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1969893, upper bound: 1339.1954254
time: 0.94 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1990508, upper bound: 1339.1954045
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -243.8700562, 219.9268188, -270.5464783, 244.2955170, -488.1655884, 490.4732666
1: -932.3855591, 794.1821899, -1035.5146484, 883.1575928, -1815.5430908, 1829.6967773
2: -494.3555603, 822.3479614, -548.4223633, 911.6830444, -1406.0385742, 1370.7702637
3: -852.5319824, 726.4732056, -945.4089966, 807.7874756, -1660.3193359, 1671.8819580
4: -621.8061523, 834.5756226, -689.7091675, 925.9588623, -1547.7650146, 1524.2847900

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1963433, upper bound: 1339.1924414
time: 1.00 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1986785, upper bound: 1339.1924322
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -242.3203888, 218.9593048, -263.0179443, 237.8081360, -480.1285400, 481.9772339
1: -925.8214111, 790.2940063, -1005.7678223, 859.2022095, -1785.0236816, 1796.0615234
2: -492.0736389, 819.5138550, -534.2224121, 888.6787109, -1380.7521973, 1353.7363281
3: -846.3413696, 722.7150269, -918.2740479, 785.6621094, -1632.0034180, 1640.9890137
4: -617.6457520, 831.6823120, -670.5506592, 902.1471558, -1519.7929688, 1502.2327881

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2055909, upper bound: 1339.2001638
time: 0.94 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2074059, upper bound: 1339.2008825
time: 1.28 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -243.3440704, 219.8070831, -266.7506409, 240.9954376, -484.3395081, 486.5577393
1: -929.8668823, 793.5015259, -1020.7149048, 871.2656250, -1801.1323242, 1814.2164307
2: -494.1090698, 822.6834106, -541.1517334, 899.5463867, -1393.6555176, 1363.8349609
3: -849.9916992, 725.6152954, -931.6386719, 796.7051392, -1646.6967773, 1657.2537842
4: -620.2249756, 834.8746948, -679.9631958, 913.4835205, -1533.7083740, 1514.8378906

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2055934, upper bound: 1339.2006644
time: 1.00 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2083430, upper bound: 1339.2009829
time: 0.95 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -242.9771881, 219.4935913, -261.5817871, 236.4643402, -479.4414978, 481.0753784
1: -928.4030762, 792.1785278, -1000.2421265, 854.3334961, -1782.7363281, 1792.4205322
2: -493.2718811, 821.5209961, -531.2681274, 883.6828613, -1376.9547119, 1352.7890625
3: -848.6797485, 724.5898438, -913.2939453, 781.2739868, -1629.9536133, 1637.8837891
4: -619.2756348, 833.7263184, -666.8989868, 897.0922852, -1516.3679199, 1500.6252441

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2033770, upper bound: 1339.1999929
time: 1.24 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2065653, upper bound: 1339.2007562
time: 1.20 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -244.0845795, 220.4189606, -265.4915161, 239.8234253, -483.9079895, 485.9104614
1: -932.7545776, 795.6243286, -1015.8863525, 866.9948120, -1799.7493896, 1811.5106201
2: -495.5066833, 825.0528564, -538.5645752, 895.2042847, -1390.7109375, 1363.6171875
3: -852.6159668, 727.7230225, -927.2700195, 792.8600464, -1645.4759521, 1654.9927979
4: -622.0684204, 837.2686157, -676.7504272, 909.0875244, -1531.1560059, 1514.0190430

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2045430, upper bound: 1339.2000294
time: 0.88 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2065464, upper bound: 1339.2007562
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -253.2254944, 228.5409851, -267.7835999, 241.8783569, -495.1038513, 496.3245544
1: -968.3505249, 824.9183960, -1024.6252441, 874.5639038, -1842.9143066, 1849.5433350
2: -514.1798096, 854.8079224, -542.8139648, 902.2751465, -1416.4549561, 1397.6217041
3: -885.1092529, 754.5663452, -935.6332397, 799.5233765, -1684.6325684, 1690.1994629
4: -645.6370239, 867.6465454, -682.7790527, 916.5136108, -1562.1506348, 1550.4255371

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1951401, upper bound: 1339.1952381
time: 1.16 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1922867, upper bound: 1339.1938424
time: 1.08 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -252.1390533, 227.5181885, -270.4276123, 244.2439270, -496.3829346, 497.9457397
1: -964.1863403, 821.2012329, -1035.0113525, 883.0403442, -1847.2266846, 1856.2126465
2: -511.9313049, 851.0175781, -548.2401733, 911.3908691, -1423.3221436, 1399.2574463
3: -881.3708496, 751.2064209, -944.9061890, 807.5344238, -1688.9049072, 1696.1125488
4: -642.9020996, 863.8019409, -689.3927002, 925.6832886, -1568.5854492, 1553.1942139

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1947312, upper bound: 1339.1924391
time: 1.08 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1922497, upper bound: 1339.1921525
time: 1.11 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -253.3698883, 228.3346252, -233.7981873, 210.8998566, -464.2697449, 462.1328125
1: -969.0567627, 823.7134399, -894.1912842, 760.6896362, -1729.7463379, 1717.9046631
2: -514.7943115, 854.2956543, -474.8912659, 788.8543091, -1303.6485596, 1329.1868896
3: -885.2359009, 753.5993042, -816.6022949, 696.0002441, -1581.2360840, 1570.2015381
4: -645.4916382, 866.5679321, -595.6613770, 799.9719849, -1445.4632568, 1462.2292480

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2032115, upper bound: 1339.2013440
time: 1.20 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2040727, upper bound: 1339.2019961
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -253.3698883, 228.3346252, -252.9692535, 228.1091156, -481.4790039, 481.3038940
1: -969.0567627, 823.7134399, -967.9032593, 822.6008911, -1791.6577148, 1791.6166992
2: -514.7943115, 854.2956543, -514.4145508, 853.3258057, -1368.1201172, 1368.7099609
3: -885.2359009, 753.5993042, -883.7941284, 752.7252808, -1637.9609375, 1637.3934326
4: -645.4916382, 866.5679321, -644.3358765, 865.5685425, -1511.0601807, 1510.9035645

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2032115, upper bound: 1339.2013440
time: 0.97 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2040727, upper bound: 1339.2019961
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -253.4208527, 228.5604553, -235.3833313, 212.5204468, -465.9412842, 463.9437256
1: -969.2923584, 825.0028687, -899.5517578, 766.8540039, -1736.1463623, 1724.5546875
2: -514.6920166, 855.2584229, -477.8874817, 795.0321655, -1309.7241211, 1333.1458740
3: -885.4770508, 754.9356079, -822.2465210, 701.4919434, -1586.9689941, 1577.1821289
4: -645.5200806, 867.8516846, -600.0542603, 806.5374146, -1452.0574951, 1467.9058838

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1826869, upper bound: 1339.1792387
time: 0.98 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1837120, upper bound: 1339.1797102
time: 1.05 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -252.5502625, 227.7652130, -242.7365875, 219.0948639, -471.6451111, 470.5018005
1: -965.9306030, 822.2180176, -928.5375977, 790.0133057, -1755.9438477, 1750.7556152
2: -512.8073120, 852.2657471, -493.2957458, 819.4750366, -1332.2823486, 1345.5615234
3: -882.4575195, 752.3942261, -848.2214355, 722.8792725, -1605.3367920, 1600.6156006
4: -643.3617554, 864.8040161, -618.6840820, 831.2706299, -1474.6323242, 1483.4879150

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.2015799, upper bound: 1339.1979091
time: 0.96 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.2022210, upper bound: 1339.1979171
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -249.2323761, 224.6605835, -248.2728882, 224.5424042, -473.7747803, 472.9334717
1: -952.9570312, 810.7577515, -948.1972046, 810.0259399, -1762.9829102, 1758.9547119
2: -506.3888855, 840.3183594, -506.0500793, 839.4983521, -1345.8872070, 1346.3682861
3: -870.7844849, 741.3861694, -866.4413452, 740.2279663, -1611.0124512, 1607.8273926
4: -635.2048950, 852.5380249, -632.8560181, 851.8345947, -1487.0395508, 1485.3940430

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2079086, upper bound: 1339.2038829
time: 1.02 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2079086, upper bound: 1339.2038829
time: 0.98 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -250.1212158, 225.4537048, -252.7064209, 228.5966949, -478.7178955, 478.1600952
1: -956.3539429, 813.5806274, -965.2999268, 824.8466797, -1781.2006836, 1778.8804932
2: -508.1991577, 843.3220825, -515.0302124, 854.9891968, -1363.1881104, 1358.3522949
3: -873.8541870, 744.0204468, -882.0025024, 753.8028564, -1627.6568604, 1626.0227051
4: -637.4530029, 855.5505981, -644.0845947, 867.3687134, -1504.8217773, 1499.6352539

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2079086, upper bound: 1339.2047031
time: 1.06 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2079086, upper bound: 1339.2047031
time: 0.99 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -251.6357880, 226.9920807, -250.6635895, 226.7819824, -478.4177856, 477.6556702
1: -962.1596069, 819.6027222, -957.1589355, 818.3115845, -1780.4711914, 1776.7617188
2: -510.9943237, 849.2695923, -510.5927124, 847.9129639, -1358.9072266, 1359.8623047
3: -879.2036743, 749.7212524, -874.9111328, 747.9197998, -1627.1235352, 1624.6323242
4: -641.1964111, 861.8271484, -639.0721436, 860.4395142, -1501.6359863, 1500.8992920

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1826663, upper bound: 1339.1783952
time: 0.98 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.1837064, upper bound: 1339.1784714
time: 0.95 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -251.0597076, 226.4570312, -258.2972717, 233.8223572, -484.8820496, 484.7543030
1: -959.9390259, 817.7318726, -986.9648438, 843.1685181, -1803.1075439, 1804.6967773
2: -509.6677246, 847.2441406, -526.6754150, 874.3708496, -1384.0384521, 1373.9195557
3: -877.2258301, 748.0224609, -901.6687012, 770.5718384, -1647.7976074, 1649.6911621
4: -639.7980957, 859.7620850, -658.4803467, 887.0756226, -1526.8736572, 1518.2424316

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2086693, upper bound: 1339.2016510
time: 1.03 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2087440, upper bound: 1339.2016500
time: 1.07 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -258.6899719, 233.6429138, -234.1498871, 211.2692261, -469.9591980, 467.7927856
1: -989.4574585, 842.6859131, -895.4569092, 762.1353149, -1751.5926514, 1738.1428223
2: -526.9299927, 874.6837769, -475.6562805, 790.0891113, -1317.0189209, 1350.3400879
3: -903.6679077, 770.6726685, -817.7363281, 697.2136841, -1600.8814697, 1588.4089355
4: -658.9655151, 887.1510010, -596.5746460, 801.2897339, -1460.2552490, 1483.7255859

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.2018523, upper bound: 1339.2013527
time: 0.93 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2040702, upper bound: 1339.2019961
time: 7.76 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -258.6899719, 233.6429138, -253.4971619, 228.6444702, -487.3344116, 487.1400757
1: -989.4574585, 842.6859131, -969.8408813, 824.6759033, -1814.1333008, 1812.5268555
2: -526.9299927, 874.6837769, -515.5205688, 855.0827637, -1382.0126953, 1390.2042236
3: -903.6679077, 770.6726685, -885.5531616, 754.4237671, -1658.0916748, 1656.2258301
4: -658.9655151, 887.1510010, -645.7116089, 867.3970337, -1526.3625488, 1532.8625488

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.2018523, upper bound: 1339.2013527
time: 0.91 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2040702, upper bound: 1339.2019961
time: 1.51 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -257.5954590, 232.7528229, -242.6504364, 219.0623322, -476.6577759, 475.4032593
1: -985.1260376, 839.6812744, -928.0777588, 790.0657349, -1775.1917725, 1767.7590332
2: -524.1746216, 870.7495728, -493.1550293, 819.1812744, -1343.3555908, 1363.9045410
3: -900.1088257, 767.9828491, -847.8090820, 722.7798462, -1622.8885498, 1615.7918701
4: -656.3240967, 883.6450806, -618.4792480, 831.0738525, -1487.3979492, 1502.1242676

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.2015845, upper bound: 1339.1979115
time: 1.24 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1339.2022019, upper bound: 1339.1979143
time: 1.14 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -255.0001068, 230.3896790, -248.3362885, 224.6332245, -479.6333313, 478.7258911
1: -975.0990601, 831.1264038, -948.3922119, 810.4515991, -1785.5502930, 1779.5184326
2: -519.4765625, 862.1364136, -506.2021790, 839.7554932, -1359.2319336, 1368.3385010
3: -890.7535400, 759.7539673, -866.5919800, 740.4993896, -1631.2528076, 1626.3459473
4: -649.8060913, 874.5958252, -633.0174561, 852.1314697, -1501.9375000, 1507.6131592

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2039395, upper bound: 1339.2038699
time: 1.00 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2039395, upper bound: 1339.2038699
time: 1.15 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -255.8878784, 231.1830597, -252.6792145, 228.6184998, -484.5062866, 483.8622742
1: -978.5148315, 833.9705200, -965.1738892, 824.9671631, -1803.4819336, 1799.1441650
2: -521.2815552, 865.1430664, -515.0261841, 855.0454102, -1376.3269043, 1380.1690674
3: -893.8613892, 762.4009399, -881.8342285, 753.8541870, -1647.7154541, 1644.2351074
4: -652.0499878, 877.6340942, -644.0071411, 867.4429932, -1519.4927979, 1521.6412354

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2040902, upper bound: 1339.2047031
time: 0.97 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1339.2040902, upper bound: 1339.2047031
time: 1.12 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.36 + 416.89 = 420.25 seconds
