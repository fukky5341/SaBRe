## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 1)
Time budget: 420 seconds
Split limit: 100
Threshold: 9177.495374428498


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3932.7343750, 6716.3618164, -3932.7343750, 6716.3618164, -10649.0947266, 10649.0947266)
1: (-677.7418213, 871.9747925, -677.7418213, 871.9747925, -1549.7165527, 1549.7165527)
2: (-520.9997559, 1149.9592285, -520.9997559, 1149.9592285, -1670.9588623, 1670.9589844)
3: (-550.2935181, 1535.1552734, -550.2935181, 1535.1552734, -2085.4487305, 2085.4487305)
4: (-452.5996704, 1453.1818848, -452.5996704, 1453.1818848, -1905.7814941, 1905.7814941)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.30 + 2.08 = 4.38 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -9177.5871503, upper bound: 9177.5871501

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5870657, upper bound: 9177.5869584
time: 0.69 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5869607, upper bound: 9177.5869605
time: 0.74 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.62 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 1.62
Output dim: 0, lower bound: -9177.5870657, upper bound: 9177.5869584
NS_B2, status: Status.UNKNOWN, split count: 1, time: 1.62
Output dim: 0, lower bound: -9177.5869607, upper bound: 9177.5869605

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -3870.2919922, 6624.3813477, -3826.1330566, 6560.2539062, -10430.5439453, 10450.5126953
1: -668.8604736, 859.6868286, -662.6802368, 851.1157227, -1519.9761963, 1522.3668213
2: -513.6199341, 1133.1857910, -508.4560547, 1121.4425049, -1635.0622559, 1641.6418457
3: -541.9356689, 1512.6304932, -536.0247803, 1496.7990723, -2038.7347412, 2048.6552734
4: -446.0426636, 1431.9000244, -441.4363403, 1417.0001221, -1863.0427246, 1873.3361816

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5869585, upper bound: 9177.5869585
time: 0.78 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5869585, upper bound: 9177.5869575
time: 0.86 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -3891.6020508, 6644.8266602, -4385.5283203, 7404.3608398, -11295.9619141, 11030.3525391
1: -670.5660400, 862.7411499, -744.4470825, 963.5292969, -1634.0953369, 1607.1881104
2: -515.4447021, 1137.9428711, -576.2767944, 1273.2928467, -1788.7375488, 1714.2194824
3: -544.4777222, 1518.9515381, -611.6740723, 1700.2182617, -2244.6960449, 2130.6247559
4: -447.7479248, 1438.0065918, -501.7661438, 1609.7399902, -2057.4877930, 1939.7727051

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5869574, upper bound: 9177.5869607
time: 0.75 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5869585, upper bound: 9177.5869605
time: 0.76 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.84 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 3.84
Output dim: 0, lower bound: -9177.5869585, upper bound: 9177.5869585
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 3.84
Output dim: 0, lower bound: -9177.5869585, upper bound: 9177.5869575
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 3.84
Output dim: 0, lower bound: -9177.5869574, upper bound: 9177.5869607
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 3.84
Output dim: 0, lower bound: -9177.5869585, upper bound: 9177.5869605

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -3826.1330566, 6560.2539062, -3826.1330566, 6560.2539062, -10386.3867188, 10386.3847656
1: -662.6802368, 851.1157227, -662.6802368, 851.1157227, -1513.7956543, 1513.7956543
2: -508.4560547, 1121.4425049, -508.4560547, 1121.4425049, -1629.8983154, 1629.8983154
3: -536.0247803, 1496.7990723, -536.0247803, 1496.7990723, -2032.8238525, 2032.8238525
4: -441.4363403, 1417.0001221, -441.4363403, 1417.0001221, -1858.4364014, 1858.4364014

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_A1

### Relational analysis result of NS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5867641, upper bound: 9177.5862405
time: 0.70 seconds

## Relational analysis of NS_B1_A1_A2

### Relational analysis result of NS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5870646, upper bound: 9177.5869593
time: 0.72 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -4385.5283203, 7404.3608398, -3826.1330566, 6560.2539062, -10945.7812500, 11230.4931641
1: -744.4470825, 963.5292969, -662.6802368, 851.1157227, -1595.5625000, 1626.2093506
2: -576.2767944, 1273.2928467, -508.4560547, 1121.4425049, -1697.7192383, 1781.7487793
3: -611.6740723, 1700.2182617, -536.0247803, 1496.7990723, -2108.4726562, 2236.2431641
4: -501.7661438, 1609.7399902, -441.4363403, 1417.0001221, -1918.7662354, 2051.1762695

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5866493, upper bound: 9177.5865243
time: 0.71 seconds

## Relational analysis of NS_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5868893, upper bound: 9177.5868048
time: 0.74 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -3826.1330566, 6560.2539062, -4385.5283203, 7404.3608398, -11230.4921875, 10945.7822266
1: -662.6802368, 851.1157227, -744.4470825, 963.5292969, -1626.2093506, 1595.5625000
2: -508.4560547, 1121.4425049, -576.2767944, 1273.2928467, -1781.7487793, 1697.7192383
3: -536.0247803, 1496.7990723, -611.6740723, 1700.2182617, -2236.2431641, 2108.4726562
4: -441.4363403, 1417.0001221, -501.7661438, 1609.7399902, -2051.1760254, 1918.7662354

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A1_A1

### Relational analysis result of NS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5860148, upper bound: 9177.5848771
time: 0.91 seconds

## Relational analysis of NS_B2_A1_A2

### Relational analysis result of NS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5860147, upper bound: 9177.5868042
time: 0.78 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -4385.5283203, 7404.3608398, -4385.5283203, 7404.3608398, -11789.8886719, 11789.8886719
1: -744.4470825, 963.5292969, -744.4470825, 963.5292969, -1707.9761963, 1707.9761963
2: -576.2767944, 1273.2928467, -576.2767944, 1273.2928467, -1849.5695801, 1849.5695801
3: -611.6740723, 1700.2182617, -611.6740723, 1700.2182617, -2311.8918457, 2311.8918457
4: -501.7661438, 1609.7399902, -501.7661438, 1609.7399902, -2111.5058594, 2111.5058594

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_A1

### Relational analysis result of NS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5867258, upper bound: 9177.5862377
time: 0.62 seconds

## Relational analysis of NS_B2_A2_A2

### Relational analysis result of NS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5869575, upper bound: 9177.5869585
time: 0.80 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.74 seconds
NS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 0, lower bound: -9177.5867641, upper bound: 9177.5862405
NS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 0, lower bound: -9177.5870646, upper bound: 9177.5869593
NS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 0, lower bound: -9177.5866493, upper bound: 9177.5865243
NS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 0, lower bound: -9177.5868893, upper bound: 9177.5868048
NS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 0, lower bound: -9177.5860148, upper bound: 9177.5848771
NS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 0, lower bound: -9177.5860147, upper bound: 9177.5868042
NS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 0, lower bound: -9177.5867258, upper bound: 9177.5862377
NS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 0, lower bound: -9177.5869575, upper bound: 9177.5869585

## BFS NS instance: NS_B1_A1_A1

### Backsubstitution after applying NS history:
0: -3419.2734375, 5866.9785156, -3626.4020996, 6227.1342773, -9646.4082031, 9493.3798828
1: -594.6650391, 760.9846191, -631.3825684, 807.2408447, -1401.9057617, 1392.3671875
2: -454.5809937, 1000.4165649, -482.5325317, 1062.3837891, -1516.9648438, 1482.9489746
3: -479.7897034, 1337.7355957, -508.7452698, 1419.2744141, -1899.0640869, 1846.4808350
4: -394.3653870, 1264.5106201, -418.7107239, 1342.1673584, -1736.5325928, 1683.2211914

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_A1_B1

### Relational analysis result of NS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5866372, upper bound: 9177.5866371
time: 0.79 seconds

## Relational analysis of NS_B1_A1_A1_B2

### Relational analysis result of NS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5866372, upper bound: 9177.5866510
time: 0.69 seconds

## BFS NS instance: NS_B1_A1_A2

### Backsubstitution after applying NS history:
0: -3782.4965820, 6487.3129883, -3805.5568848, 6525.7817383, -10308.2773438, 10292.8701172
1: -655.3565674, 841.7345581, -659.2196655, 846.6872559, -1502.0438232, 1500.9542236
2: -502.7701721, 1109.3443604, -505.7691650, 1115.7314453, -1618.5012207, 1615.1131592
3: -529.9553833, 1480.2968750, -533.1580200, 1489.0080566, -2018.9633789, 2013.4548340
4: -436.4317932, 1401.7626953, -439.0714417, 1409.8116455, -1846.2432861, 1840.8339844

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_A2_B1

### Relational analysis result of NS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5866511, upper bound: 9177.5869143
time: 0.63 seconds

## Relational analysis of NS_B1_A1_A2_B2

### Relational analysis result of NS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5866511, upper bound: 9177.5871077
time: 0.80 seconds

## BFS NS instance: NS_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4314.7138672, 7284.3662109, -3637.4792480, 6237.5742188, -10552.2832031, 10921.8457031
1: -732.4514771, 948.0684204, -630.7015381, 809.6492310, -1542.1007080, 1578.7698975
2: -566.8430176, 1253.2901611, -483.0583801, 1067.7856445, -1634.6284180, 1736.3483887
3: -601.7001343, 1672.9372559, -509.2463989, 1423.3089600, -2025.0090332, 2182.1833496
4: -493.5930786, 1584.5740967, -419.3038330, 1349.3654785, -1842.9584961, 2003.8779297

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_B1_B1

### Relational analysis result of NS_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5858031, upper bound: 9177.5843909
time: 0.72 seconds

## Relational analysis of NS_B1_A2_B1_B2

### Relational analysis result of NS_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5847074, upper bound: 9177.5843526
time: 0.81 seconds

## BFS NS instance: NS_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4352.5092773, 7350.7880859, -3762.2365723, 6453.9511719, -10806.4589844, 11113.0244141
1: -739.1738281, 956.5723877, -652.2570801, 837.4876099, -1576.6613770, 1608.8294678
2: -572.0924683, 1264.1850586, -500.2144775, 1103.5531006, -1675.6455078, 1764.3995361
3: -607.0861206, 1687.7155762, -527.0872192, 1472.2095947, -2079.2956543, 2214.8024902
4: -498.0954285, 1598.2993164, -434.2152405, 1394.3890381, -1892.4844971, 2032.5145264

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A2_B2_A1

### Relational analysis result of NS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5860588, upper bound: 9177.5848733
time: 0.70 seconds

## Relational analysis of NS_B1_A2_B2_A2

### Relational analysis result of NS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5860588, upper bound: 9177.5868038
time: 0.77 seconds

## BFS NS instance: NS_B2_A1_A1

### Backsubstitution after applying NS history:
0: -3637.4792480, 6237.5742188, -4314.7138672, 7284.3662109, -10921.8457031, 10552.2832031
1: -630.7015381, 809.6492310, -732.4514771, 948.0684204, -1578.7698975, 1542.1007080
2: -483.0583801, 1067.7856445, -566.8430176, 1253.2901611, -1736.3483887, 1634.6284180
3: -509.2463989, 1423.3089600, -601.7001343, 1672.9372559, -2182.1833496, 2025.0090332
4: -419.3038330, 1349.3654785, -493.5930786, 1584.5740967, -2003.8779297, 1842.9584961

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_A1_A1

### Relational analysis result of NS_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5837496, upper bound: 9177.5858032
time: 0.76 seconds

## Relational analysis of NS_B2_A1_A1_A2

### Relational analysis result of NS_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5843528, upper bound: 9177.5847074
time: 0.68 seconds

## BFS NS instance: NS_B2_A1_A2

### Backsubstitution after applying NS history:
0: -3762.2365723, 6453.9511719, -4352.5092773, 7350.7880859, -11113.0244141, 10806.4589844
1: -652.2570801, 837.4876099, -739.1738281, 956.5723877, -1608.8294678, 1576.6613770
2: -500.2144775, 1103.5531006, -572.0924683, 1264.1850586, -1764.3995361, 1675.6455078
3: -527.0872192, 1472.2095947, -607.0861206, 1687.7155762, -2214.8022461, 2079.2956543
4: -434.2152405, 1394.3890381, -498.0954285, 1598.2993164, -2032.5145264, 1892.4844971

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A1_A2_B1

### Relational analysis result of NS_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5848734, upper bound: 9177.5860588
time: 0.71 seconds

## Relational analysis of NS_B2_A1_A2_B2

### Relational analysis result of NS_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5848734, upper bound: 9177.5868884
time: 0.78 seconds

## BFS NS instance: NS_B2_A2_A1

### Backsubstitution after applying NS history:
0: -3956.6308594, 6682.0385742, -4189.6010742, 7081.1464844, -11037.7744141, 10871.6376953
1: -673.9899292, 869.5902100, -714.2515259, 921.0062866, -1594.9960938, 1583.8417969
2: -520.2813110, 1146.5751953, -551.2490234, 1215.8016357, -1736.0827637, 1697.8242188
3: -552.4536133, 1533.6242676, -584.9230957, 1624.6793213, -2177.1325684, 2118.5468750
4: -452.7416687, 1450.0175781, -479.7768555, 1536.8662109, -1989.6079102, 1929.7943115

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_A1_B1

### Relational analysis result of NS_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5862259, upper bound: 9177.5862259
time: 0.79 seconds

## Relational analysis of NS_B2_A2_A1_B2

### Relational analysis result of NS_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5862259, upper bound: 9177.5862377
time: 0.66 seconds

## BFS NS instance: NS_B2_A2_A2

### Backsubstitution after applying NS history:
0: -4338.7412109, 7325.5932617, -4363.0146484, 7366.0600586, -11704.7988281, 11688.6074219
1: -736.5399170, 953.3863525, -740.5908203, 958.6038208, -1695.1437988, 1693.9771729
2: -570.1271973, 1260.3269043, -573.2844238, 1267.0220947, -1837.1492920, 1833.6112061
3: -605.2031860, 1682.3829346, -608.5590210, 1691.5783691, -2296.7814941, 2290.9416504
4: -496.3843079, 1593.3913574, -499.1583252, 1601.8359375, -2098.2202148, 2092.5493164

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_A2_B1

### Relational analysis result of NS_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5862377, upper bound: 9177.5867258
time: 0.82 seconds

## Relational analysis of NS_B2_A2_A2_B2

### Relational analysis result of NS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5862377, upper bound: 9177.5869572
time: 0.92 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.11 seconds
NS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 0, lower bound: -9177.5866372, upper bound: 9177.5866371
NS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 0, lower bound: -9177.5866372, upper bound: 9177.5866510
NS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 0, lower bound: -9177.5866511, upper bound: 9177.5869143
NS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 0, lower bound: -9177.5866511, upper bound: 9177.5871077
NS_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 0, lower bound: -9177.5858031, upper bound: 9177.5843909
NS_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 0, lower bound: -9177.5847074, upper bound: 9177.5843526
NS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 0, lower bound: -9177.5860588, upper bound: 9177.5848733
NS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 0, lower bound: -9177.5860588, upper bound: 9177.5868038
NS_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 0, lower bound: -9177.5837496, upper bound: 9177.5858032
NS_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 0, lower bound: -9177.5843528, upper bound: 9177.5847074
NS_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 0, lower bound: -9177.5848734, upper bound: 9177.5860588
NS_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 0, lower bound: -9177.5848734, upper bound: 9177.5868884
NS_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 0, lower bound: -9177.5862259, upper bound: 9177.5862259
NS_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 0, lower bound: -9177.5862259, upper bound: 9177.5862377
NS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 0, lower bound: -9177.5862377, upper bound: 9177.5867258
NS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 0, lower bound: -9177.5862377, upper bound: 9177.5869572

## BFS NS instance: NS_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -3419.2734375, 5866.9785156, -3419.2734375, 5866.9785156, -9286.2509766, 9286.2509766
1: -594.6650391, 760.9846191, -594.6650391, 760.9846191, -1355.6495361, 1355.6495361
2: -454.5809937, 1000.4165649, -454.5809937, 1000.4165649, -1454.9975586, 1454.9975586
3: -479.7897034, 1337.7355957, -479.7897034, 1337.7355957, -1817.5252686, 1817.5252686
4: -394.3653870, 1264.5106201, -394.3653870, 1264.5106201, -1658.8758545, 1658.8758545

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_A1_B1_B1

### Relational analysis result of NS_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5863855, upper bound: 9177.5864244
time: 0.75 seconds

## Relational analysis of NS_B1_A1_A1_B1_B2

### Relational analysis result of NS_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5864307, upper bound: 9177.5864307
time: 0.72 seconds

## BFS NS instance: NS_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -3419.2734375, 5866.9785156, -3782.4965820, 6487.3129883, -9906.5849609, 9649.4746094
1: -594.6650391, 760.9846191, -655.3565674, 841.7345581, -1436.3992920, 1416.3411865
2: -454.5809937, 1000.4165649, -502.7701721, 1109.3443604, -1563.9252930, 1503.1862793
3: -479.7897034, 1337.7355957, -529.9553833, 1480.2968750, -1960.0865479, 1867.6909180
4: -394.3653870, 1264.5106201, -436.4317932, 1401.7626953, -1796.1279297, 1700.9422607

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_A1_B2_A1

### Relational analysis result of NS_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5864244, upper bound: 9177.5864016
time: 0.69 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2

### Relational analysis result of NS_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5864307, upper bound: 9177.5864428
time: 0.84 seconds

## BFS NS instance: NS_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -3782.4965820, 6487.3129883, -3419.2734375, 5866.9785156, -9649.4746094, 9906.5859375
1: -655.3565674, 841.7345581, -594.6650391, 760.9846191, -1416.3411865, 1436.3992920
2: -502.7701721, 1109.3443604, -454.5809937, 1000.4165649, -1503.1862793, 1563.9252930
3: -529.9553833, 1480.2968750, -479.7897034, 1337.7355957, -1867.6909180, 1960.0865479
4: -436.4317932, 1401.7626953, -394.3653870, 1264.5106201, -1700.9422607, 1796.1279297

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_A2_B1_B1

### Relational analysis result of NS_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5864016, upper bound: 9177.5867515
time: 0.66 seconds

## Relational analysis of NS_B1_A1_A2_B1_B2

### Relational analysis result of NS_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5864428, upper bound: 9177.5867662
time: 0.82 seconds

## BFS NS instance: NS_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -3782.4965820, 6487.3129883, -3782.4965820, 6487.3129883, -10269.8085938, 10269.8085938
1: -655.3565674, 841.7345581, -655.3565674, 841.7345581, -1497.0910645, 1497.0910645
2: -502.7701721, 1109.3443604, -502.7701721, 1109.3443604, -1612.1141357, 1612.1141357
3: -529.9553833, 1480.2968750, -529.9553833, 1480.2968750, -2010.2521973, 2010.2521973
4: -436.4317932, 1401.7626953, -436.4317932, 1401.7626953, -1838.1943359, 1838.1943359

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_A2_B2_B1

### Relational analysis result of NS_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5864015, upper bound: 9177.5869289
time: 0.79 seconds

## Relational analysis of NS_B1_A1_A2_B2_B2

### Relational analysis result of NS_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5864427, upper bound: 9177.5869303
time: 0.72 seconds

## BFS NS instance: NS_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -4295.8867188, 7254.8945312, -3564.8771973, 6125.3183594, -10421.2041016, 10819.7705078
1: -729.7844238, 944.1680298, -620.5459595, 794.6858521, -1524.4702148, 1564.7139893
2: -564.5508423, 1248.0701904, -474.2496033, 1047.8067627, -1612.3576660, 1722.3197021
3: -599.0924072, 1665.8725586, -499.2705994, 1396.1896973, -1995.2821045, 2165.1430664
4: -491.5385437, 1577.9185791, -411.4421692, 1323.8457031, -1815.3842773, 1989.3604736

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_B1_B1_A1

### Relational analysis result of NS_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5847074, upper bound: 9177.5843528
time: 0.76 seconds

## Relational analysis of NS_B1_A2_B1_B1_A2

### Relational analysis result of NS_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5847074, upper bound: 9177.5843528
time: 0.65 seconds

## BFS NS instance: NS_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -4193.4477539, 7076.5517578, -3774.4667969, 6440.8750000, -10634.3212891, 10851.0185547
1: -710.7531738, 921.0485229, -650.2651367, 836.0762939, -1546.8293457, 1571.3133545
2: -550.6177979, 1218.4934082, -498.9023132, 1104.6992188, -1655.3167725, 1717.3953857
3: -585.0877686, 1625.8133545, -528.8224487, 1471.8944092, -2056.9821777, 2154.6352539
4: -479.7448730, 1540.6059570, -433.8751831, 1395.3730469, -1875.1179199, 1974.4810791

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_B1_B2_A1

### Relational analysis result of NS_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5847074, upper bound: 9177.5843528
time: 0.82 seconds

## Relational analysis of NS_B1_A2_B1_B2_A2

### Relational analysis result of NS_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5847074, upper bound: 9177.5843528
time: 0.68 seconds

## BFS NS instance: NS_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4167.0712891, 7029.9321289, -3762.2365723, 6453.9511719, -10621.0224609, 10792.1689453
1: -707.4539185, 915.6871338, -652.2570801, 837.4876099, -1544.9415283, 1567.9442139
2: -546.8971558, 1210.7711182, -500.2144775, 1103.5531006, -1650.4501953, 1710.9854736
3: -580.6928711, 1615.3194580, -527.0872192, 1472.2095947, -2052.9023438, 2142.4062500
4: -476.0477295, 1530.8754883, -434.2152405, 1394.3890381, -1870.4366455, 1965.0906982

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_B2_A1_A1

### Relational analysis result of NS_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5837198, upper bound: 9177.5846039
time: 0.78 seconds

## Relational analysis of NS_B1_A2_B2_A1_A2

### Relational analysis result of NS_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5837091, upper bound: 9177.5843138
time: 0.62 seconds

## BFS NS instance: NS_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4319.4291992, 7297.4916992, -3762.2365723, 6453.9511719, -10773.3789062, 11059.7265625
1: -733.9296265, 949.6303711, -652.2570801, 837.4876099, -1571.4172363, 1601.8874512
2: -567.9162598, 1255.1153564, -500.2144775, 1103.5531006, -1671.4692383, 1755.3298340
3: -602.4892578, 1675.2551270, -527.0872192, 1472.2095947, -2074.6987305, 2202.3417969
4: -494.4281616, 1586.9332275, -434.2152405, 1394.3890381, -1888.8171387, 2021.1484375

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_B2_A2_B1

### Relational analysis result of NS_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5856755, upper bound: 9177.5843941
time: 0.65 seconds

## Relational analysis of NS_B1_A2_B2_A2_B2

### Relational analysis result of NS_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5837091, upper bound: 9177.5843258
time: 0.67 seconds

## BFS NS instance: NS_B2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -3564.8771973, 6125.3183594, -4295.8867188, 7254.8945312, -10819.7705078, 10421.2041016
1: -620.5459595, 794.6858521, -729.7844238, 944.1680298, -1564.7139893, 1524.4702148
2: -474.2496033, 1047.8067627, -564.5508423, 1248.0701904, -1722.3197021, 1612.3576660
3: -499.2705994, 1396.1896973, -599.0924072, 1665.8725586, -2165.1430664, 1995.2821045
4: -411.4421692, 1323.8457031, -491.5385437, 1577.9185791, -1989.3604736, 1815.3842773

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_A1_A1_B1

### Relational analysis result of NS_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5843526, upper bound: 9177.5847072
time: 0.70 seconds

## Relational analysis of NS_B2_A1_A1_A1_B2

### Relational analysis result of NS_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5843528, upper bound: 9177.5847074
time: 0.69 seconds

## BFS NS instance: NS_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -3774.4667969, 6440.8750000, -4193.4477539, 7076.5517578, -10851.0185547, 10634.3212891
1: -650.2651367, 836.0762939, -710.7531738, 921.0485229, -1571.3133545, 1546.8292236
2: -498.9023132, 1104.6992188, -550.6177979, 1218.4934082, -1717.3953857, 1655.3167725
3: -528.8224487, 1471.8944092, -585.0877686, 1625.8133545, -2154.6352539, 2056.9821777
4: -433.8751831, 1395.3730469, -479.7448730, 1540.6059570, -1974.4810791, 1875.1179199

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_A1_A2_B1

### Relational analysis result of NS_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5837187, upper bound: 9177.5847072
time: 0.70 seconds

## Relational analysis of NS_B2_A1_A1_A2_B2

### Relational analysis result of NS_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5843528, upper bound: 9177.5847074
time: 0.66 seconds

## BFS NS instance: NS_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -3762.2365723, 6453.9511719, -4167.0712891, 7029.9321289, -10792.1689453, 10621.0224609
1: -652.2570801, 837.4876099, -707.4539185, 915.6871338, -1567.9442139, 1544.9415283
2: -500.2144775, 1103.5531006, -546.8971558, 1210.7711182, -1710.9854736, 1650.4501953
3: -527.0872192, 1472.2095947, -580.6928711, 1615.3194580, -2142.4062500, 2052.9023438
4: -434.2152405, 1394.3890381, -476.0477295, 1530.8754883, -1965.0906982, 1870.4366455

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_A2_B1_B1

### Relational analysis result of NS_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5846039, upper bound: 9177.5837197
time: 0.79 seconds

## Relational analysis of NS_B2_A1_A2_B1_B2

### Relational analysis result of NS_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5843137, upper bound: 9177.5837091
time: 0.73 seconds

## BFS NS instance: NS_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -3762.2365723, 6453.9511719, -4319.4291992, 7297.4916992, -11059.7265625, 10773.3789062
1: -652.2570801, 837.4876099, -733.9296265, 949.6303711, -1601.8874512, 1571.4172363
2: -500.2144775, 1103.5531006, -567.9162598, 1255.1153564, -1755.3298340, 1671.4692383
3: -527.0872192, 1472.2095947, -602.4892578, 1675.2551270, -2202.3415527, 2074.6987305
4: -434.2152405, 1394.3890381, -494.4281616, 1586.9332275, -2021.1484375, 1888.8171387

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_A2_B2_A1

### Relational analysis result of NS_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5843848, upper bound: 9177.5856754
time: 0.80 seconds

## Relational analysis of NS_B2_A1_A2_B2_A2

### Relational analysis result of NS_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5843138, upper bound: 9177.5837091
time: 0.67 seconds

## BFS NS instance: NS_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -3956.6308594, 6682.0385742, -3956.6308594, 6682.0385742, -10638.6679688, 10638.6679688
1: -673.9899292, 869.5902100, -673.9899292, 869.5902100, -1543.5799561, 1543.5799561
2: -520.2813110, 1146.5751953, -520.2813110, 1146.5751953, -1666.8563232, 1666.8563232
3: -552.4536133, 1533.6242676, -552.4536133, 1533.6242676, -2086.0778809, 2086.0778809
4: -452.7416687, 1450.0175781, -452.7416687, 1450.0175781, -1902.7592773, 1902.7592773

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_A2_A1_B1_B1

### Relational analysis result of NS_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5862048, upper bound: 9177.5860731
time: 0.71 seconds

## Relational analysis of NS_B2_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_A1_B1_A1

### Relational analysis result of NS_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5856966, upper bound: 9177.5856966
time: 0.84 seconds

## Relational analysis of NS_B2_A2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B2_A2_A1_B1_A1

### Relational analysis result of NS_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5855223, upper bound: 9177.5849019
time: 0.63 seconds

## Relational analysis of NS_B2_A2_A1_B1_A2

### Relational analysis result of NS_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5855738, upper bound: 9177.5855738
time: 0.76 seconds

## BFS NS instance: NS_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -3956.6308594, 6682.0385742, -4338.7412109, 7325.5932617, -11282.2246094, 11020.7783203
1: -673.9899292, 869.5902100, -736.5399170, 953.3863525, -1627.3762207, 1606.1301270
2: -520.2813110, 1146.5751953, -570.1271973, 1260.3269043, -1780.6080322, 1716.7022705
3: -552.4536133, 1533.6242676, -605.2031860, 1682.3829346, -2234.8364258, 2138.8273926
4: -452.7416687, 1450.0175781, -496.3843079, 1593.3913574, -2046.1330566, 1946.4018555

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B2_A2_A1_B2_A1

### Relational analysis result of NS_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5860709, upper bound: 9177.5861806
time: 0.86 seconds

## Relational analysis of NS_B2_A2_A1_B2_A2

### Relational analysis result of NS_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5860710, upper bound: 9177.5860860
time: 0.82 seconds

## BFS NS instance: NS_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -4338.7412109, 7325.5932617, -3956.6308594, 6682.0385742, -11020.7783203, 11282.2246094
1: -736.5399170, 953.3863525, -673.9899292, 869.5902100, -1606.1301270, 1627.3762207
2: -570.1271973, 1260.3269043, -520.2813110, 1146.5751953, -1716.7022705, 1780.6080322
3: -605.2031860, 1682.3829346, -552.4536133, 1533.6242676, -2138.8273926, 2234.8364258
4: -496.3843079, 1593.3913574, -452.7416687, 1450.0175781, -1946.4018555, 2046.1330566

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_A2_A2_B1_B1

### Relational analysis result of NS_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5862189, upper bound: 9177.5867082
time: 0.79 seconds

## Relational analysis of NS_B2_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_A2_B1_B1

### Relational analysis result of NS_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5861212, upper bound: 9177.5866081
time: 0.78 seconds

## Relational analysis of NS_B2_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B2_A2_A2_B1_B1

### Relational analysis result of NS_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5861678, upper bound: 9177.5866444
time: 0.68 seconds

## Relational analysis of NS_B2_A2_A2_B1_B2

### Relational analysis result of NS_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5862343, upper bound: 9177.5866573
time: 0.81 seconds

## BFS NS instance: NS_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -4338.7412109, 7325.5932617, -4338.7412109, 7325.5932617, -11664.3339844, 11664.3339844
1: -736.5399170, 953.3863525, -736.5399170, 953.3863525, -1689.9262695, 1689.9262695
2: -570.1271973, 1260.3269043, -570.1271973, 1260.3269043, -1830.4539795, 1830.4539795
3: -605.2031860, 1682.3829346, -605.2031860, 1682.3829346, -2287.5859375, 2287.5859375
4: -496.3843079, 1593.3913574, -496.3843079, 1593.3913574, -2089.7756348, 2089.7756348

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_A2_A2_B2_B1

### Relational analysis result of NS_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5862188, upper bound: 9177.5869568
time: 0.80 seconds

## Relational analysis of NS_B2_A2_A2_B2_B2

### Relational analysis result of NS_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5860826, upper bound: 9177.5869577
time: 0.80 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.44 seconds
NS_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.44
Output dim: 0, lower bound: -9177.5863855, upper bound: 9177.5864244
NS_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.44
Output dim: 0, lower bound: -9177.5864307, upper bound: 9177.5864307
NS_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.44
Output dim: 0, lower bound: -9177.5864244, upper bound: 9177.5864016
NS_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.44
Output dim: 0, lower bound: -9177.5864307, upper bound: 9177.5864428
NS_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.44
Output dim: 0, lower bound: -9177.5864016, upper bound: 9177.5867515
NS_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.44
Output dim: 0, lower bound: -9177.5864428, upper bound: 9177.5867662
NS_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.44
Output dim: 0, lower bound: -9177.5864015, upper bound: 9177.5869289
NS_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.44
Output dim: 0, lower bound: -9177.5864427, upper bound: 9177.5869303
NS_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.44
Output dim: 0, lower bound: -9177.5847074, upper bound: 9177.5843528
NS_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.44
Output dim: 0, lower bound: -9177.5847074, upper bound: 9177.5843528
NS_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.44
Output dim: 0, lower bound: -9177.5847074, upper bound: 9177.5843528
NS_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.44
Output dim: 0, lower bound: -9177.5847074, upper bound: 9177.5843528
NS_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.44
Output dim: 0, lower bound: -9177.5837198, upper bound: 9177.5846039
NS_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.44
Output dim: 0, lower bound: -9177.5837091, upper bound: 9177.5843138
NS_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.44
Output dim: 0, lower bound: -9177.5856755, upper bound: 9177.5843941
NS_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.44
Output dim: 0, lower bound: -9177.5837091, upper bound: 9177.5843258
NS_B2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.44
Output dim: 0, lower bound: -9177.5843526, upper bound: 9177.5847072
NS_B2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.44
Output dim: 0, lower bound: -9177.5843528, upper bound: 9177.5847074
NS_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.44
Output dim: 0, lower bound: -9177.5837187, upper bound: 9177.5847072
NS_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.44
Output dim: 0, lower bound: -9177.5843528, upper bound: 9177.5847074
NS_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.44
Output dim: 0, lower bound: -9177.5846039, upper bound: 9177.5837197
NS_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.44
Output dim: 0, lower bound: -9177.5843137, upper bound: 9177.5837091
NS_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.44
Output dim: 0, lower bound: -9177.5843848, upper bound: 9177.5856754
NS_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.44
Output dim: 0, lower bound: -9177.5843138, upper bound: 9177.5837091
NS_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.44
Output dim: 0, lower bound: -9177.5855223, upper bound: 9177.5849019
NS_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.44
Output dim: 0, lower bound: -9177.5855738, upper bound: 9177.5855738
NS_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.44
Output dim: 0, lower bound: -9177.5860709, upper bound: 9177.5861806
NS_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.44
Output dim: 0, lower bound: -9177.5860710, upper bound: 9177.5860860
NS_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.44
Output dim: 0, lower bound: -9177.5861678, upper bound: 9177.5866444
NS_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.44
Output dim: 0, lower bound: -9177.5862343, upper bound: 9177.5866573
NS_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.44
Output dim: 0, lower bound: -9177.5862188, upper bound: 9177.5869568
NS_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.44
Output dim: 0, lower bound: -9177.5860826, upper bound: 9177.5869577

## BFS NS instance: NS_B1_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -3353.6672363, 5755.7026367, -3258.4345703, 5593.8608398, -8947.5263672, 9014.1357422
1: -583.5056763, 746.6770020, -567.6415405, 725.8853760, -1309.3906250, 1314.3183594
2: -445.8350220, 981.6754150, -433.0787048, 955.2138672, -1401.0488281, 1414.7541504
3: -470.5097351, 1312.5133057, -456.7520752, 1275.2490234, -1745.7587891, 1769.2653809
4: -386.7492981, 1241.0672607, -375.5470581, 1207.4464111, -1594.1954346, 1616.6142578

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_A1_B1_B1_A1

### Relational analysis result of NS_B1_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5863818, upper bound: 9177.5863818
time: 0.74 seconds

## Relational analysis of NS_B1_A1_A1_B1_B1_A2

### Relational analysis result of NS_B1_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5863818, upper bound: 9177.5864245
time: 0.78 seconds

## BFS NS instance: NS_B1_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -3387.3781738, 5814.1035156, -3355.0410156, 5760.6503906, -9148.0283203, 9169.1445312
1: -589.5175781, 754.1390381, -584.3161011, 747.2489624, -1336.7664795, 1338.4547119
2: -450.4539795, 991.4584351, -446.2889099, 982.4076538, -1432.8615723, 1437.7471924
3: -475.2728882, 1325.4921875, -470.7170105, 1313.0791016, -1788.3518066, 1796.2089844
4: -390.7600708, 1253.1448975, -387.1147156, 1241.6763916, -1632.4364014, 1640.2593994

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_A1_B1_B2_A1

### Relational analysis result of NS_B1_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5864245, upper bound: 9177.5863856
time: 0.77 seconds

## Relational analysis of NS_B1_A1_A1_B1_B2_A2

### Relational analysis result of NS_B1_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5864245, upper bound: 9177.5864307
time: 0.83 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3258.4345703, 5593.8608398, -3701.6093750, 6348.5961914, -9607.0302734, 9295.4687500
1: -567.6415405, 725.8853760, -641.4600830, 823.9725342, -1391.6138916, 1367.3453369
2: -433.0787048, 955.2138672, -491.8999023, 1086.2087402, -1519.2874756, 1447.1137695
3: -456.7520752, 1275.2490234, -518.5279541, 1449.0197754, -1905.7718506, 1793.7768555
4: -375.5470581, 1207.4464111, -426.9882812, 1372.7623291, -1748.3093262, 1634.4345703

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B1_A1_A1_B2_A1_B1

### Relational analysis result of NS_B1_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5865064, upper bound: 9177.5859285
time: 0.78 seconds

## Relational analysis of NS_B1_A1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_A1_B2_A1_B1

### Relational analysis result of NS_B1_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5864699, upper bound: 9177.5855995
time: 0.78 seconds

## Relational analysis of NS_B1_A1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_A1_B2_A1_B1

### Relational analysis result of NS_B1_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5867440, upper bound: 9177.5864011
time: 0.83 seconds

## Relational analysis of NS_B1_A1_A1_B2_A1_B2

### Relational analysis result of NS_B1_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5867440, upper bound: 9177.5864017
time: 0.64 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3355.0410156, 5760.6503906, -3750.7812500, 6434.5434570, -9789.5839844, 9511.4316406
1: -584.3161011, 747.2489624, -650.1859741, 834.9531860, -1419.2690430, 1397.4349365
2: -446.2889099, 982.4076538, -498.6741028, 1100.4490967, -1546.7379150, 1481.0817871
3: -470.7170105, 1313.0791016, -525.5321655, 1468.0762939, -1938.7930908, 1838.6113281
4: -387.1147156, 1241.6763916, -432.8558044, 1390.5177002, -1777.6322021, 1674.5322266

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_A1_B2_A2_B1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5867557, upper bound: 9177.5864401
time: 0.71 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5867557, upper bound: 9177.5864428
time: 0.82 seconds

## BFS NS instance: NS_B1_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -3701.6093750, 6348.5961914, -3258.4345703, 5593.8608398, -9295.4687500, 9607.0302734
1: -641.4600830, 823.9725342, -567.6415405, 725.8853760, -1367.3450928, 1391.6138916
2: -491.8999023, 1086.2087402, -433.0787048, 955.2138672, -1447.1137695, 1519.2874756
3: -518.5279541, 1449.0197754, -456.7520752, 1275.2490234, -1793.7768555, 1905.7718506
4: -426.9882812, 1372.7623291, -375.5470581, 1207.4464111, -1634.4345703, 1748.3093262

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_A1_A2_B1_B1_A1

### Relational analysis result of NS_B1_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5859291, upper bound: 9177.5865064
time: 0.67 seconds

## Relational analysis of NS_B1_A1_A2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_A2_B1_B1_A1

### Relational analysis result of NS_B1_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5855995, upper bound: 9177.5864699
time: 0.79 seconds

## Relational analysis of NS_B1_A1_A2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_A2_B1_B1_A1

### Relational analysis result of NS_B1_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5864011, upper bound: 9177.5867440
time: 0.65 seconds

## Relational analysis of NS_B1_A1_A2_B1_B1_A2

### Relational analysis result of NS_B1_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5864011, upper bound: 9177.5867515
time: 0.76 seconds

## BFS NS instance: NS_B1_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -3750.7812500, 6434.5434570, -3355.0410156, 5760.6503906, -9511.4316406, 9789.5839844
1: -650.1859741, 834.9531860, -584.3161011, 747.2489624, -1397.4349365, 1419.2690430
2: -498.6741028, 1100.4490967, -446.2889099, 982.4076538, -1481.0817871, 1546.7379150
3: -525.5321655, 1468.0762939, -470.7170105, 1313.0791016, -1838.6113281, 1938.7930908
4: -432.8558044, 1390.5177002, -387.1147156, 1241.6763916, -1674.5322266, 1777.6322021

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_A2_B1_B2_A1

### Relational analysis result of NS_B1_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5864416, upper bound: 9177.5867558
time: 0.72 seconds

## Relational analysis of NS_B1_A1_A2_B1_B2_A2

### Relational analysis result of NS_B1_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5864416, upper bound: 9177.5867662
time: 0.70 seconds

## BFS NS instance: NS_B1_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -3701.6093750, 6348.5961914, -3594.7451172, 6165.7958984, -9867.4052734, 9943.3417969
1: -641.4600830, 823.9725342, -623.5287476, 800.4345703, -1441.8945312, 1447.5012207
2: -491.8999023, 1086.2087402, -477.4615173, 1055.9501953, -1547.8499756, 1563.6701660
3: -518.5279541, 1449.0197754, -503.2953491, 1407.0817871, -1925.6097412, 1952.3151855
4: -426.9882812, 1372.7623291, -414.3929443, 1334.4486084, -1761.4368896, 1787.1552734

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_A2_B2_B1_A1

### Relational analysis result of NS_B1_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5869294, upper bound: 9177.5869273
time: 0.70 seconds

## Relational analysis of NS_B1_A1_A2_B2_B1_A2

### Relational analysis result of NS_B1_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5869305, upper bound: 9177.5869279
time: 0.74 seconds

## BFS NS instance: NS_B1_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -3750.7812500, 6434.5434570, -3718.3562012, 6380.8208008, -10131.6015625, 10152.8994141
1: -650.1859741, 834.9531860, -644.9185791, 828.0537720, -1478.2396240, 1479.8718262
2: -498.6741028, 1100.4490967, -494.5073547, 1091.3853760, -1590.0594482, 1594.9562988
3: -525.5321655, 1468.0762939, -521.0017090, 1455.6260986, -1981.1582031, 1989.0780029
4: -432.8558044, 1390.5177002, -429.2016602, 1379.0643311, -1811.9201660, 1819.7193604

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_A2_B2_B2_A1

### Relational analysis result of NS_B1_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5869301, upper bound: 9177.5869307
time: 0.76 seconds

## Relational analysis of NS_B1_A1_A2_B2_B2_A2

### Relational analysis result of NS_B1_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5864245, upper bound: 9177.5869312
time: 0.77 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -4245.9082031, 7175.5502930, -3564.8771973, 6125.3183594, -10371.2265625, 10740.4277344
1: -722.5538940, 933.7323608, -620.5459595, 794.6858521, -1517.2397461, 1554.2781982
2: -558.3963623, 1234.0672607, -474.2496033, 1047.8067627, -1606.2031250, 1708.3167725
3: -592.1253052, 1646.9715576, -499.2705994, 1396.1896973, -1988.3149414, 2146.2421875
4: -486.0226440, 1560.1060791, -411.4421692, 1323.8457031, -1809.8684082, 1971.5478516

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_B1_B1_A1_B1

### Relational analysis result of NS_B1_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5858018, upper bound: 9177.5843873
time: 0.68 seconds

## Relational analysis of NS_B1_A2_B1_B1_A1_B2

### Relational analysis result of NS_B1_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5857266, upper bound: 9177.5843907
time: 0.77 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -4524.1123047, 7588.9736328, -3564.8771973, 6125.3183594, -10649.4306641, 11153.8505859
1: -761.8208008, 987.5803223, -620.5459595, 794.6858521, -1556.5065918, 1608.1262207
2: -591.0237427, 1306.4896240, -474.2496033, 1047.8067627, -1638.8305664, 1780.7392578
3: -631.2873535, 1746.7325439, -499.2705994, 1396.1896973, -2027.4770508, 2246.0031738
4: -515.9533081, 1651.4349365, -411.4421692, 1323.8457031, -1839.7990723, 2062.8767090

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B1_B1_A2_A1

### Relational analysis result of NS_B1_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5855597, upper bound: 9177.5841284
time: 0.65 seconds

## Relational analysis of NS_B1_A2_B1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_B1_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5858018, upper bound: 9177.5843874
time: 0.65 seconds

## Relational analysis of NS_B1_A2_B1_B1_A2_B2

### Relational analysis result of NS_B1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5857266, upper bound: 9177.5843909
time: 0.64 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -4240.7631836, 7165.9531250, -3774.4667969, 6440.8750000, -10681.6386719, 10940.4199219
1: -721.5800781, 932.5192261, -650.2651367, 836.0762939, -1557.6562500, 1582.7843018
2: -557.6519165, 1232.5651855, -498.9023132, 1104.6992188, -1662.3510742, 1731.4674072
3: -591.3888550, 1644.9099121, -528.8224487, 1471.8944092, -2063.2829590, 2173.7321777
4: -485.3816833, 1558.2187500, -433.8751831, 1395.3730469, -1880.7547607, 1992.0937500

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A2_B1_B2_A1_A1

### Relational analysis result of NS_B1_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5847074, upper bound: 9177.5843487
time: 0.70 seconds

## Relational analysis of NS_B1_A2_B1_B2_A1_A2

### Relational analysis result of NS_B1_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5847074, upper bound: 9177.5843528
time: 0.75 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -4524.1123047, 7588.9736328, -3774.4667969, 6440.8750000, -10964.9873047, 11363.4404297
1: -761.8208008, 987.5803223, -650.2651367, 836.0762939, -1597.8969727, 1637.8453369
2: -591.0237427, 1306.4896240, -498.9023132, 1104.6992188, -1695.7229004, 1805.3917236
3: -631.2873535, 1746.7325439, -528.8224487, 1471.8944092, -2103.1816406, 2275.5549316
4: -515.9533081, 1651.4349365, -433.8751831, 1395.3730469, -1911.3264160, 2085.3098145

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B1_A2_B1_B2_A2_B1

### Relational analysis result of NS_B1_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5844940, upper bound: 9177.5842843
time: 0.66 seconds

## Relational analysis of NS_B1_A2_B1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A2_B1_B2_A2_A1

### Relational analysis result of NS_B1_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5847072, upper bound: 9177.5843487
time: 0.76 seconds

## Relational analysis of NS_B1_A2_B1_B2_A2_A2

### Relational analysis result of NS_B1_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5847074, upper bound: 9177.5843526
time: 0.74 seconds

## BFS NS instance: NS_B1_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -4098.0551758, 6921.7778320, -3742.6865234, 6423.9726562, -10522.0273438, 10664.4628906
1: -697.6532593, 901.4195557, -649.5549927, 833.4841309, -1531.1373291, 1550.9746094
2: -538.4706421, 1191.6674805, -497.8577271, 1098.2226562, -1636.6932373, 1689.5250244
3: -571.1142578, 1589.4495850, -524.3881226, 1464.9470215, -2036.0612793, 2113.8376465
4: -468.5844421, 1506.5167236, -432.1103516, 1387.5627441, -1856.1469727, 1938.6270752

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_B2_A1_A1_B1

### Relational analysis result of NS_B1_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5837091, upper bound: 9177.5843473
time: 0.69 seconds

## Relational analysis of NS_B1_A2_B2_A1_A1_B2

### Relational analysis result of NS_B1_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5837091, upper bound: 9177.5843473
time: 0.72 seconds

## BFS NS instance: NS_B1_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -4403.7841797, 7381.6059570, -3642.0019531, 6248.7792969, -10652.5634766, 11023.6074219
1: -741.1290894, 960.8952026, -630.7551880, 810.8252563, -1551.9543457, 1591.6502686
2: -574.6804199, 1271.7470703, -484.2105103, 1069.1707764, -1643.8510742, 1755.9573975
3: -614.3331909, 1699.8081055, -510.6122131, 1425.7053223, -2040.0385742, 2210.4204102
4: -501.7614441, 1607.4949951, -420.5372925, 1350.9495850, -1852.7109375, 2028.0318604

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_B2_A1_A2_B1

### Relational analysis result of NS_B1_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5837091, upper bound: 9177.5843473
time: 0.91 seconds

## Relational analysis of NS_B1_A2_B2_A1_A2_B2

### Relational analysis result of NS_B1_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5837091, upper bound: 9177.5843473
time: 0.60 seconds

## BFS NS instance: NS_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4300.5737305, 7268.0522461, -3691.4973145, 6344.2763672, -10644.8496094, 10959.5498047
1: -731.2630005, 945.7294922, -642.3369751, 822.8883057, -1554.1513672, 1588.0664062
2: -565.6238403, 1249.8869629, -491.6228638, 1084.1145020, -1649.7382812, 1741.5096436
3: -599.8804321, 1668.1988525, -517.2888794, 1445.7645264, -2045.6448975, 2185.4875488
4: -492.3726196, 1580.2690430, -426.5450439, 1369.5129395, -1861.8854980, 2006.8140869

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5837137, upper bound: 9177.5843259
time: 0.71 seconds

## Relational analysis of NS_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5837137, upper bound: 9177.5843258
time: 0.73 seconds

## BFS NS instance: NS_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4197.5810547, 7088.6684570, -3879.2111816, 6621.0996094, -10818.6796875, 10967.8798828
1: -712.0889282, 922.4439087, -668.2537842, 859.3580933, -1571.4468994, 1590.6973877
2: -551.5709839, 1220.1309814, -513.1363525, 1134.5122070, -1686.0832520, 1733.2670898
3: -585.7563477, 1627.9536133, -543.7461548, 1512.9168701, -2098.6728516, 2171.6997070
4: -480.4827576, 1542.7480469, -446.2893677, 1433.3032227, -1913.7860107, 1989.0373535

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5837137, upper bound: 9177.5843258
time: 0.68 seconds

## Relational analysis of NS_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5837136, upper bound: 9177.5843258
time: 0.62 seconds

## BFS NS instance: NS_B2_A1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -3564.8771973, 6125.3183594, -4245.9082031, 7175.5502930, -10740.4277344, 10371.2265625
1: -620.5459595, 794.6858521, -722.5538940, 933.7323608, -1554.2781982, 1517.2397461
2: -474.2496033, 1047.8067627, -558.3963623, 1234.0672607, -1708.3167725, 1606.2031250
3: -499.2705994, 1396.1896973, -592.1253052, 1646.9715576, -2146.2421875, 1988.3149414
4: -411.4421692, 1323.8457031, -486.0226440, 1560.1060791, -1971.5478516, 1809.8684082

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B2_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B2_A1_A1_A1_B1_A1

### Relational analysis result of NS_B2_A1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5837496, upper bound: 9177.5858018
time: 0.73 seconds

## Relational analysis of NS_B2_A1_A1_A1_B1_A2

### Relational analysis result of NS_B2_A1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5843909, upper bound: 9177.5857264
time: 0.64 seconds

## BFS NS instance: NS_B2_A1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -3564.8771973, 6125.3183594, -4524.1123047, 7588.9736328, -11153.8505859, 10649.4306641
1: -620.5459595, 794.6858521, -761.8208008, 987.5803223, -1608.1262207, 1556.5065918
2: -474.2496033, 1047.8067627, -591.0237427, 1306.4896240, -1780.7392578, 1638.8305664
3: -499.2705994, 1396.1896973, -631.2873535, 1746.7325439, -2246.0031738, 2027.4770508
4: -411.4421692, 1323.8457031, -515.9533081, 1651.4349365, -2062.8764648, 1839.7990723

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B2_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B2_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B2_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B2_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_A1_A1_B2_B1

### Relational analysis result of NS_B2_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5841285, upper bound: 9177.5855597
time: 0.69 seconds

## Relational analysis of NS_B2_A1_A1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B2_A1_A1_A1_B2_A1

### Relational analysis result of NS_B2_A1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5837496, upper bound: 9177.5858017
time: 0.69 seconds

## Relational analysis of NS_B2_A1_A1_A1_B2_A2

### Relational analysis result of NS_B2_A1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5843907, upper bound: 9177.5857266
time: 0.67 seconds

## BFS NS instance: NS_B2_A1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -3774.4667969, 6440.8750000, -4240.7631836, 7165.9531250, -10940.4199219, 10681.6386719
1: -650.2651367, 836.0762939, -721.5800781, 932.5192261, -1582.7843018, 1557.6562500
2: -498.9023132, 1104.6992188, -557.6519165, 1232.5651855, -1731.4674072, 1662.3510742
3: -528.8224487, 1471.8944092, -591.3888550, 1644.9099121, -2173.7321777, 2063.2829590
4: -433.8751831, 1395.3730469, -485.3816833, 1558.2187500, -1992.0937500, 1880.7547607

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B2_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B2_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B2_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A1_A1_A2_B1_B1

### Relational analysis result of NS_B2_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5843486, upper bound: 9177.5847074
time: 0.67 seconds

## Relational analysis of NS_B2_A1_A1_A2_B1_B2

### Relational analysis result of NS_B2_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5843487, upper bound: 9177.5847072
time: 0.70 seconds

## BFS NS instance: NS_B2_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -3774.4667969, 6440.8750000, -4524.1123047, 7588.9736328, -11363.4404297, 10964.9873047
1: -650.2651367, 836.0762939, -761.8208008, 987.5803223, -1637.8453369, 1597.8969727
2: -498.9023132, 1104.6992188, -591.0237427, 1306.4896240, -1805.3917236, 1695.7229004
3: -528.8224487, 1471.8944092, -631.2873535, 1746.7325439, -2275.5549316, 2103.1816406
4: -433.8751831, 1395.3730469, -515.9533081, 1651.4349365, -2085.3095703, 1911.3264160

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B2_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B2_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B2_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B2_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B2_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_A1_A2_B2_A1

### Relational analysis result of NS_B2_A1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5842844, upper bound: 9177.5844940
time: 0.61 seconds

## Relational analysis of NS_B2_A1_A1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B2_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A1_A1_A2_B2_B1

### Relational analysis result of NS_B2_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5843487, upper bound: 9177.5847072
time: 0.65 seconds

## Relational analysis of NS_B2_A1_A1_A2_B2_B2

### Relational analysis result of NS_B2_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5843487, upper bound: 9177.5847074
time: 0.75 seconds

## BFS NS instance: NS_B2_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -3742.6865234, 6423.9726562, -4098.0551758, 6921.7778320, -10664.4628906, 10522.0273438
1: -649.5549927, 833.4841309, -697.6532593, 901.4195557, -1550.9746094, 1531.1373291
2: -497.8577271, 1098.2226562, -538.4706421, 1191.6674805, -1689.5250244, 1636.6932373
3: -524.3881226, 1464.9470215, -571.1142578, 1589.4495850, -2113.8376465, 2036.0612793
4: -432.1103516, 1387.5627441, -468.5844421, 1506.5167236, -1938.6270752, 1856.1469727

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_A2_B1_B1_A1

### Relational analysis result of NS_B2_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5843473, upper bound: 9177.5837089
time: 0.77 seconds

## Relational analysis of NS_B2_A1_A2_B1_B1_A2

### Relational analysis result of NS_B2_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5843473, upper bound: 9177.5837089
time: 0.60 seconds

## BFS NS instance: NS_B2_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -3642.0019531, 6248.7792969, -4403.7841797, 7381.6059570, -11023.6074219, 10652.5634766
1: -630.7551880, 810.8252563, -741.1290894, 960.8952026, -1591.6502686, 1551.9543457
2: -484.2105103, 1069.1707764, -574.6804199, 1271.7470703, -1755.9573975, 1643.8510742
3: -510.6122131, 1425.7053223, -614.3331909, 1699.8081055, -2210.4204102, 2040.0385742
4: -420.5372925, 1350.9495850, -501.7614441, 1607.4949951, -2028.0318604, 1852.7109375

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_A2_B1_B2_A1

### Relational analysis result of NS_B2_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5843473, upper bound: 9177.5837091
time: 0.71 seconds

## Relational analysis of NS_B2_A1_A2_B1_B2_A2

### Relational analysis result of NS_B2_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5843473, upper bound: 9177.5837091
time: 0.67 seconds

## BFS NS instance: NS_B2_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3691.4973145, 6344.2763672, -4300.5737305, 7268.0522461, -10959.5498047, 10644.8496094
1: -642.3369751, 822.8883057, -731.2630005, 945.7294922, -1588.0664062, 1554.1513672
2: -491.6228638, 1084.1145020, -565.6238403, 1249.8869629, -1741.5097656, 1649.7382812
3: -517.2888794, 1445.7645264, -599.8804321, 1668.1988525, -2185.4875488, 2045.6448975
4: -426.5450439, 1369.5129395, -492.3726196, 1580.2690430, -2006.8139648, 1861.8854980

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_A2_B2_A1_B1

### Relational analysis result of NS_B2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5843138, upper bound: 9177.5837089
time: 0.70 seconds

## Relational analysis of NS_B2_A1_A2_B2_A1_B2

### Relational analysis result of NS_B2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5843137, upper bound: 9177.5837091
time: 0.71 seconds

## BFS NS instance: NS_B2_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3879.2111816, 6621.0996094, -4197.5810547, 7088.6684570, -10967.8798828, 10818.6796875
1: -668.2537842, 859.3580933, -712.0889282, 922.4439087, -1590.6973877, 1571.4468994
2: -513.1363525, 1134.5122070, -551.5709839, 1220.1309814, -1733.2670898, 1686.0832520
3: -543.7461548, 1512.9168701, -585.7563477, 1627.9536133, -2171.6997070, 2098.6728516
4: -446.2893677, 1433.3032227, -480.4827576, 1542.7480469, -1989.0373535, 1913.7860107

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_A2_B2_A2_B1

### Relational analysis result of NS_B2_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5843138, upper bound: 9177.5837091
time: 0.76 seconds

## Relational analysis of NS_B2_A1_A2_B2_A2_B2

### Relational analysis result of NS_B2_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5843138, upper bound: 9177.5837091
time: 0.74 seconds

## BFS NS instance: NS_B2_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3619.3708496, 6132.3159180, -3737.6589355, 6329.3354492, -9948.7060547, 9869.9726562
1: -621.3000488, 796.8502197, -640.7119141, 822.7136230, -1444.0136719, 1437.5621338
2: -477.8272095, 1042.8724365, -492.9964905, 1079.5183105, -1557.3454590, 1535.8688965
3: -506.4156799, 1402.6732178, -522.6006470, 1449.0993652, -1955.5150146, 1925.2738037
4: -415.8298035, 1318.9224854, -428.9478149, 1365.2078857, -1781.0377197, 1747.8702393

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B2_A2_A1_B1_A1_B1

### Relational analysis result of NS_B2_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5848781, upper bound: 9177.5848736
time: 0.65 seconds

## Relational analysis of NS_B2_A2_A1_B1_A1_B2

### Relational analysis result of NS_B2_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5848782, upper bound: 9177.5849018
time: 0.74 seconds

## BFS NS instance: NS_B2_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3920.7329102, 6624.2680664, -3938.1872559, 6652.1914062, -10572.9238281, 10562.4521484
1: -668.3213501, 861.9279175, -671.0596313, 865.6365356, -1533.9578857, 1532.9874268
2: -515.7199707, 1136.2677002, -517.9267578, 1141.2653809, -1656.9852295, 1654.1944580
3: -547.5558472, 1520.0731201, -549.9362793, 1526.6461182, -2074.2019043, 2070.0092773
4: -448.7582092, 1437.0153809, -450.6876526, 1443.3225098, -1892.0806885, 1887.7030029

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B2_A2_A1_B1_A2_B1

### Relational analysis result of NS_B2_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5848950, upper bound: 9177.5855138
time: 0.66 seconds

## Relational analysis of NS_B2_A2_A1_B1_A2_B2

### Relational analysis result of NS_B2_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5848953, upper bound: 9177.5855736
time: 0.90 seconds

## BFS NS instance: NS_B2_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3856.9982910, 6528.7128906, -4273.9458008, 7223.0747070, -11080.0683594, 10802.6582031
1: -659.4873657, 849.3746338, -726.8363037, 939.9461670, -1599.4334717, 1576.2108154
2: -508.2416687, 1118.2854004, -562.0994873, 1241.5368652, -1749.7784424, 1680.3848877
3: -538.6727295, 1496.7456055, -596.2538452, 1658.0072021, -2196.6799316, 2092.9992676
4: -442.0471191, 1414.0279541, -489.2870789, 1569.5883789, -2011.6354980, 1903.3150635

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B2_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B2_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B2_A2_A1_B2_A1_A1

### Relational analysis result of NS_B2_A2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5860305, upper bound: 9177.5845715
time: 0.70 seconds

## Relational analysis of NS_B2_A2_A1_B2_A1_A2

### Relational analysis result of NS_B2_A2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5863832, upper bound: 9177.5854957
time: 0.77 seconds

## BFS NS instance: NS_B2_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3987.9838867, 6757.2719727, -4214.4765625, 7122.0458984, -11110.0292969, 10971.7480469
1: -683.1896362, 878.6084595, -716.2374268, 926.8139038, -1610.0032959, 1594.8459473
2: -526.3194580, 1155.0715332, -554.1749878, 1223.9682617, -1750.2877197, 1709.2463379
3: -557.2305298, 1547.7009277, -587.8128662, 1635.0529785, -2192.2834473, 2135.5136719
4: -458.2667847, 1460.7049561, -482.3509216, 1547.7399902, -2006.0065918, 1943.0559082

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B2_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_A2_A1_B2_A2_B1

### Relational analysis result of NS_B2_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5867056, upper bound: 9177.5860860
time: 0.71 seconds

## Relational analysis of NS_B2_A2_A1_B2_A2_B2

### Relational analysis result of NS_B2_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5867055, upper bound: 9177.5860860
time: 0.68 seconds

## BFS NS instance: NS_B2_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -4324.8212891, 7302.0825195, -3928.7187500, 6634.0698242, -10958.8906250, 11230.7998047
1: -734.1867676, 950.3847656, -669.1543579, 863.4656982, -1597.6524658, 1619.5389404
2: -568.3179932, 1256.3343506, -516.5755005, 1138.5220947, -1706.8398438, 1772.9096680
3: -603.2666016, 1677.0014648, -548.5408325, 1522.7469482, -2126.0134277, 2225.5419922
4: -494.8019104, 1588.3554688, -449.5152283, 1439.8865967, -1934.6884766, 2037.8707275

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B2_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B2_A2_A2_B1_B1_A1

### Relational analysis result of NS_B2_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5861677, upper bound: 9177.5866453
time: 0.66 seconds

## Relational analysis of NS_B2_A2_A2_B1_B1_A2

### Relational analysis result of NS_B2_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5861678, upper bound: 9177.5866444
time: 1.14 seconds

## BFS NS instance: NS_B2_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -4271.2128906, 7207.1142578, -3895.7683105, 6570.3374023, -10841.5478516, 11102.8798828
1: -725.2735596, 938.2475586, -663.9619141, 855.4249878, -1580.6984863, 1602.2093506
2: -561.0863037, 1240.2175293, -511.9817505, 1127.2185059, -1688.3044434, 1752.1990967
3: -595.7918701, 1655.5343018, -544.2388306, 1508.6151123, -2104.4067383, 2199.7731934
4: -488.4621887, 1567.8801270, -445.4405518, 1425.2968750, -1913.7590332, 2013.3206787

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B2_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_A2_B1_B2_B1

### Relational analysis result of NS_B2_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5861537, upper bound: 9177.5862272
time: 0.68 seconds

## Relational analysis of NS_B2_A2_A2_B1_B2_B2

### Relational analysis result of NS_B2_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5862260, upper bound: 9177.5862299
time: 0.81 seconds

## BFS NS instance: NS_B2_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -4273.9458008, 7223.0747070, -4237.5000000, 7165.7031250, -11439.6474609, 11460.5712891
1: -726.8363037, 939.9461670, -721.3948975, 932.4577637, -1659.2940674, 1661.3409424
2: -562.0994873, 1241.5368652, -557.6072388, 1231.0225830, -1793.1220703, 1799.1440430
3: -596.2538452, 1658.0072021, -591.2128906, 1644.3397217, -2240.5932617, 2249.2194824
4: -489.2870789, 1569.5883789, -485.3031006, 1556.2626953, -2045.5498047, 2054.8913574

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B2_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B2_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B2_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B2_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_B2_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B2_A2_A2_B2_B1_A1

### Relational analysis result of NS_B2_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5869280, upper bound: 9177.5869280
time: 0.75 seconds

## Relational analysis of NS_B2_A2_A2_B2_B1_A2

### Relational analysis result of NS_B2_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5869269, upper bound: 9177.5869568
time: 0.77 seconds

## BFS NS instance: NS_B2_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -4214.4765625, 7122.0458984, -4355.9331055, 7371.4458008, -11585.9218750, 11477.9785156
1: -716.2374268, 926.8139038, -742.8804321, 958.7864990, -1675.0239258, 1669.6943359
2: -554.1749878, 1223.9682617, -573.9114380, 1263.6345215, -1817.8093262, 1797.8796387
3: -587.8128662, 1635.0529785, -608.1095581, 1690.2895508, -2278.1022949, 2243.1623535
4: -482.3509216, 1547.7399902, -499.7807922, 1597.5814209, -2079.9323730, 2047.5206299

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A2_A2_B2_B2_A1

### Relational analysis result of NS_B2_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5868219, upper bound: 9177.5868219
time: 0.87 seconds

## Relational analysis of NS_B2_A2_A2_B2_B2_A2

### Relational analysis result of NS_B2_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5868218, upper bound: 9177.5868226
time: 0.86 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.27 seconds
NS_B1_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5863818, upper bound: 9177.5863818
NS_B1_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5863818, upper bound: 9177.5864245
NS_B1_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5864245, upper bound: 9177.5863856
NS_B1_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5864245, upper bound: 9177.5864307
NS_B1_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5867440, upper bound: 9177.5864011
NS_B1_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5867440, upper bound: 9177.5864017
NS_B1_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5867557, upper bound: 9177.5864401
NS_B1_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5867557, upper bound: 9177.5864428
NS_B1_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5864011, upper bound: 9177.5867440
NS_B1_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5864011, upper bound: 9177.5867515
NS_B1_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5864416, upper bound: 9177.5867558
NS_B1_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5864416, upper bound: 9177.5867662
NS_B1_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5869294, upper bound: 9177.5869273
NS_B1_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5869305, upper bound: 9177.5869279
NS_B1_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5869301, upper bound: 9177.5869307
NS_B1_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5864245, upper bound: 9177.5869312
NS_B1_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5858018, upper bound: 9177.5843873
NS_B1_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5857266, upper bound: 9177.5843907
NS_B1_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5858018, upper bound: 9177.5843874
NS_B1_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5857266, upper bound: 9177.5843909
NS_B1_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5847074, upper bound: 9177.5843487
NS_B1_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5847074, upper bound: 9177.5843528
NS_B1_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5847072, upper bound: 9177.5843487
NS_B1_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5847074, upper bound: 9177.5843526
NS_B1_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5837091, upper bound: 9177.5843473
NS_B1_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5837091, upper bound: 9177.5843473
NS_B1_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5837091, upper bound: 9177.5843473
NS_B1_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5837091, upper bound: 9177.5843473
NS_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5837137, upper bound: 9177.5843259
NS_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5837137, upper bound: 9177.5843258
NS_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5837137, upper bound: 9177.5843258
NS_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5837136, upper bound: 9177.5843258
NS_B2_A1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5837496, upper bound: 9177.5858018
NS_B2_A1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5843909, upper bound: 9177.5857264
NS_B2_A1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5837496, upper bound: 9177.5858017
NS_B2_A1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5843907, upper bound: 9177.5857266
NS_B2_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5843486, upper bound: 9177.5847074
NS_B2_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5843487, upper bound: 9177.5847072
NS_B2_A1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5843487, upper bound: 9177.5847072
NS_B2_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5843487, upper bound: 9177.5847074
NS_B2_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5843473, upper bound: 9177.5837089
NS_B2_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5843473, upper bound: 9177.5837089
NS_B2_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5843473, upper bound: 9177.5837091
NS_B2_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5843473, upper bound: 9177.5837091
NS_B2_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5843138, upper bound: 9177.5837089
NS_B2_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5843137, upper bound: 9177.5837091
NS_B2_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5843138, upper bound: 9177.5837091
NS_B2_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5843138, upper bound: 9177.5837091
NS_B2_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5848781, upper bound: 9177.5848736
NS_B2_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5848782, upper bound: 9177.5849018
NS_B2_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5848950, upper bound: 9177.5855138
NS_B2_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5848953, upper bound: 9177.5855736
NS_B2_A2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5860305, upper bound: 9177.5845715
NS_B2_A2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5863832, upper bound: 9177.5854957
NS_B2_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5867056, upper bound: 9177.5860860
NS_B2_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5867055, upper bound: 9177.5860860
NS_B2_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5861677, upper bound: 9177.5866453
NS_B2_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5861678, upper bound: 9177.5866444
NS_B2_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5861537, upper bound: 9177.5862272
NS_B2_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5862260, upper bound: 9177.5862299
NS_B2_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5869280, upper bound: 9177.5869280
NS_B2_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5869269, upper bound: 9177.5869568
NS_B2_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5868219, upper bound: 9177.5868219
NS_B2_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -9177.5868218, upper bound: 9177.5868226

## BFS NS instance: NS_B1_A1_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -3258.4345703, 5593.8608398, -3258.4345703, 5593.8608398, -8852.2929688, 8852.2929688
1: -567.6415405, 725.8853760, -567.6415405, 725.8853760, -1293.5268555, 1293.5268555
2: -433.0787048, 955.2138672, -433.0787048, 955.2138672, -1388.2926025, 1388.2926025
3: -456.7520752, 1275.2490234, -456.7520752, 1275.2490234, -1732.0010986, 1732.0010986
4: -375.5470581, 1207.4464111, -375.5470581, 1207.4464111, -1582.9931641, 1582.9931641

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_A1_B1_B1_A1_B1

### Relational analysis result of NS_B1_A1_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5862404, upper bound: 9177.5845992
time: 0.67 seconds

## Relational analysis of NS_B1_A1_A1_B1_B1_A1_B2

### Relational analysis result of NS_B1_A1_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5857641, upper bound: 9177.5845636
time: 0.76 seconds

## BFS NS instance: NS_B1_A1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -3351.4501953, 5753.9794922, -3258.4345703, 5593.8608398, -8945.3085938, 9012.4130859
1: -583.6404419, 746.4099121, -567.6415405, 725.8853760, -1309.5256348, 1314.0512695
2: -445.7727966, 981.3731079, -433.0787048, 955.2138672, -1400.9866943, 1414.4517822
3: -470.2115173, 1311.6547852, -456.7520752, 1275.2490234, -1745.4604492, 1768.4068604
4: -386.6682434, 1240.3789062, -375.5470581, 1207.4464111, -1594.1145020, 1615.9260254

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_A1_B1_B1_A2_A1

### Relational analysis result of NS_B1_A1_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5857685, upper bound: 9177.5862956
time: 0.72 seconds

## Relational analysis of NS_B1_A1_A1_B1_B1_A2_A2

### Relational analysis result of NS_B1_A1_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5857641, upper bound: 9177.5845637
time: 0.70 seconds

## BFS NS instance: NS_B1_A1_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -3258.4345703, 5593.8608398, -3355.0410156, 5760.6503906, -9019.0839844, 8948.9013672
1: -567.6415405, 725.8853760, -584.3161011, 747.2489624, -1314.8905029, 1310.2012939
2: -433.0787048, 955.2138672, -446.2889099, 982.4076538, -1415.4863281, 1401.5028076
3: -456.7520752, 1275.2490234, -470.7170105, 1313.0791016, -1769.8311768, 1745.9656982
4: -375.5470581, 1207.4464111, -387.1147156, 1241.6763916, -1617.2233887, 1594.5606689

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_A1_B1_B2_A1_B1

### Relational analysis result of NS_B1_A1_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5862399, upper bound: 9177.5846067
time: 0.65 seconds

## Relational analysis of NS_B1_A1_A1_B1_B2_A1_B2

### Relational analysis result of NS_B1_A1_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5845197, upper bound: 9177.5845196
time: 0.69 seconds

## BFS NS instance: NS_B1_A1_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -3355.0410156, 5760.6503906, -3355.0410156, 5760.6503906, -9115.6914062, 9115.6914062
1: -584.3161011, 747.2489624, -584.3161011, 747.2489624, -1331.5649414, 1331.5649414
2: -446.2889099, 982.4076538, -446.2889099, 982.4076538, -1428.6965332, 1428.6965332
3: -470.7170105, 1313.0791016, -470.7170105, 1313.0791016, -1783.7958984, 1783.7958984
4: -387.1147156, 1241.6763916, -387.1147156, 1241.6763916, -1628.7908936, 1628.7908936

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_A1_B1_B2_A2_B1

### Relational analysis result of NS_B1_A1_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5862399, upper bound: 9177.5845992
time: 0.73 seconds

## Relational analysis of NS_B1_A1_A1_B1_B2_A2_B2

### Relational analysis result of NS_B1_A1_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5845197, upper bound: 9177.5845197
time: 0.64 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -3258.4345703, 5593.8608398, -3594.7451172, 6165.7958984, -9424.2304688, 9188.6054688
1: -567.6415405, 725.8853760, -623.5287476, 800.4345703, -1368.0761719, 1349.4140625
2: -433.0787048, 955.2138672, -477.4615173, 1055.9501953, -1489.0286865, 1432.6752930
3: -456.7520752, 1275.2490234, -503.2953491, 1407.0817871, -1863.8338623, 1778.5443115
4: -375.5470581, 1207.4464111, -414.3929443, 1334.4486084, -1709.9956055, 1621.8392334

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A1

### Relational analysis result of NS_B1_A1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5866407, upper bound: 9177.5861164
time: 0.62 seconds

## Relational analysis of NS_B1_A1_A1_B2_A1_B1_A2

### Relational analysis result of NS_B1_A1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5866494, upper bound: 9177.5861954
time: 0.79 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3258.4345703, 5593.8608398, -3703.4902344, 6353.6816406, -9612.1132812, 9297.3515625
1: -567.6415405, 725.8853760, -642.1555176, 824.6397095, -1392.2812500, 1368.0408936
2: -433.0787048, 955.2138672, -492.4007568, 1087.1469727, -1520.2257080, 1447.6146240
3: -456.7520752, 1275.2490234, -518.8862305, 1449.7995605, -1906.5516357, 1794.1350098
4: -375.5470581, 1207.4464111, -427.3666992, 1373.7602539, -1749.3072510, 1634.8129883

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_A1

### Relational analysis result of NS_B1_A1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5866407, upper bound: 9177.5861184
time: 0.68 seconds

## Relational analysis of NS_B1_A1_A1_B2_A1_B2_A2

### Relational analysis result of NS_B1_A1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5866492, upper bound: 9177.5861982
time: 0.78 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3355.0410156, 5760.6503906, -3594.7451172, 6165.7958984, -9520.8369141, 9355.3955078
1: -584.3161011, 747.2489624, -623.5287476, 800.4345703, -1384.7504883, 1370.7777100
2: -446.2889099, 982.4076538, -477.4615173, 1055.9501953, -1502.2390137, 1459.8691406
3: -470.7170105, 1313.0791016, -503.2953491, 1407.0817871, -1877.7985840, 1816.3745117
4: -387.1147156, 1241.6763916, -414.3929443, 1334.4486084, -1721.5631104, 1656.0693359

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5862471, upper bound: 9177.5845759
time: 0.75 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5862611, upper bound: 9177.5851979
time: 0.65 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_A1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5866348, upper bound: 9177.5861470
time: 0.75 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B1_A2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5866656, upper bound: 9177.5862878
time: 0.69 seconds

## BFS NS instance: NS_B1_A1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3355.0410156, 5760.6503906, -3718.3562012, 6380.8208008, -9735.8613281, 9479.0068359
1: -584.3161011, 747.2489624, -644.9185791, 828.0537720, -1412.3696289, 1392.1674805
2: -446.2889099, 982.4076538, -494.5073547, 1091.3853760, -1537.6743164, 1476.9150391
3: -470.7170105, 1313.0791016, -521.0017090, 1455.6260986, -1926.3427734, 1834.0808105
4: -387.1147156, 1241.6763916, -429.2016602, 1379.0643311, -1766.1788330, 1670.8780518

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5862612, upper bound: 9177.5851981
time: 0.72 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5862470, upper bound: 9177.5845759
time: 0.85 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_A1

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5864788, upper bound: 9177.5862749
time: 0.73 seconds

## Relational analysis of NS_B1_A1_A1_B2_A2_B2_A2

### Relational analysis result of NS_B1_A1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5864765, upper bound: 9177.5862719
time: 0.72 seconds

## BFS NS instance: NS_B1_A1_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -3594.7451172, 6165.7958984, -3258.4345703, 5593.8608398, -9188.6054688, 9424.2304688
1: -623.5287476, 800.4345703, -567.6415405, 725.8853760, -1349.4140625, 1368.0761719
2: -477.4615173, 1055.9501953, -433.0787048, 955.2138672, -1432.6752930, 1489.0286865
3: -503.2953491, 1407.0817871, -456.7520752, 1275.2490234, -1778.5443115, 1863.8338623
4: -414.3929443, 1334.4486084, -375.5470581, 1207.4464111, -1621.8392334, 1709.9956055

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B1_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B1_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B1_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B1_A1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_A1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B1_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_B1_A1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A2_B1_B1_A1_B1

### Relational analysis result of NS_B1_A1_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5861164, upper bound: 9177.5866407
time: 0.74 seconds

## Relational analysis of NS_B1_A1_A2_B1_B1_A1_B2

### Relational analysis result of NS_B1_A1_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5861956, upper bound: 9177.5866494
time: 0.71 seconds

## BFS NS instance: NS_B1_A1_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -3703.4902344, 6353.6816406, -3258.4345703, 5593.8608398, -9297.3515625, 9612.1123047
1: -642.1555176, 824.6397095, -567.6415405, 725.8853760, -1368.0408936, 1392.2812500
2: -492.4007568, 1087.1469727, -433.0787048, 955.2138672, -1447.6146240, 1520.2257080
3: -518.8862305, 1449.7995605, -456.7520752, 1275.2490234, -1794.1350098, 1906.5516357
4: -427.3666992, 1373.7602539, -375.5470581, 1207.4464111, -1634.8129883, 1749.3073730

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B1_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B1_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B1_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_A1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_A1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B1_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B1_A1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_B1_A1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A2_B1_B1_A2_B1

### Relational analysis result of NS_B1_A1_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5861164, upper bound: 9177.5866466
time: 0.68 seconds

## Relational analysis of NS_B1_A1_A2_B1_B1_A2_B2

### Relational analysis result of NS_B1_A1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5861956, upper bound: 9177.5866576
time: 0.69 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.38 + 416.75 = 421.14 seconds
