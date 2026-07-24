## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 10002.246664433122


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5663.9658203, 5953.3359375, -5663.9658203, 5953.3359375, -11617.3007812, 11617.3007812)
1: (-599.4065552, 432.4301147, -599.4065552, 432.4301147, -1031.8366699, 1031.8366699)
2: (-973.9396973, 1117.4992676, -973.9396973, 1117.4992676, -2091.4389648, 2091.4389648)
3: (-1112.3724365, 707.8311768, -1112.3724365, 707.8311768, -1820.2034912, 1820.2034912)
4: (-842.6556396, 907.6983032, -842.6556396, 907.6983032, -1750.3540039, 1750.3540039)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.75 + 2.06 = 4.81 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -10002.3466879, upper bound: 10002.3466879

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3335084, upper bound: 10002.3349016
time: 0.81 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3386357, upper bound: 10002.3386357
time: 0.82 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.87 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.87
Output dim: 0, lower bound: -10002.3335084, upper bound: 10002.3349016
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.87
Output dim: 0, lower bound: -10002.3386357, upper bound: 10002.3386357

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -5498.2578125, 5784.8613281, -5663.9658203, 5953.3359375, -11451.5927734, 11448.8271484
1: -582.2402344, 419.9122925, -599.4065552, 432.4301147, -1014.6703491, 1019.3188477
2: -945.6491699, 1086.3648682, -973.9396973, 1117.4992676, -2063.1484375, 2060.3041992
3: -1080.5843506, 687.4306641, -1112.3724365, 707.8311768, -1788.4154053, 1799.8029785
4: -818.4217529, 882.3312988, -842.6556396, 907.6983032, -1726.1201172, 1724.9869385

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3309696, upper bound: 10002.3309696
time: 0.74 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3309696, upper bound: 10002.3349016
time: 0.79 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -6075.7607422, 6381.3359375, -5573.7333984, 5863.7387695, -11939.5000000, 11955.0693359
1: -642.4890747, 464.3061218, -590.0385742, 425.6209717, -1068.1098633, 1054.3447266
2: -1044.1705322, 1196.6938477, -958.4054565, 1100.8385010, -2145.0090332, 2155.0991211
3: -1192.0223389, 759.1086426, -1094.7279053, 696.8623657, -1888.8845215, 1853.8364258
4: -903.7961426, 972.5676880, -829.1525269, 894.1081543, -1797.9042969, 1801.7200928

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3349016, upper bound: 10002.3335084
time: 1.06 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3349016, upper bound: 10002.3386358
time: 0.73 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.59 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.59
Output dim: 0, lower bound: -10002.3309696, upper bound: 10002.3309696
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.59
Output dim: 0, lower bound: -10002.3309696, upper bound: 10002.3349016
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.59
Output dim: 0, lower bound: -10002.3349016, upper bound: 10002.3335084
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.59
Output dim: 0, lower bound: -10002.3349016, upper bound: 10002.3386358

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -5498.2578125, 5784.8613281, -5498.2578125, 5784.8613281, -11283.1191406, 11283.1191406
1: -582.2402344, 419.9122925, -582.2402344, 419.9122925, -1002.1524658, 1002.1524048
2: -945.6491699, 1086.3648682, -945.6491699, 1086.3648682, -2032.0140381, 2032.0140381
3: -1080.5843506, 687.4306641, -1080.5843506, 687.4306641, -1768.0147705, 1768.0148926
4: -818.4217529, 882.3312988, -818.4217529, 882.3312988, -1700.7530518, 1700.7530518

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9999.0899976, upper bound: 10001.7685167
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9999.0495044, upper bound: 10001.1447028
time: 0.71 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -5498.2578125, 5784.8613281, -6075.7607422, 6381.3359375, -11879.5927734, 11860.6220703
1: -582.2402344, 419.9122925, -642.4890747, 464.3061218, -1046.5463867, 1062.4011230
2: -945.6491699, 1086.3648682, -1044.1705322, 1196.6938477, -2142.3430176, 2130.5346680
3: -1080.5843506, 687.4306641, -1192.0223389, 759.1086426, -1839.6928711, 1879.4528809
4: -818.4217529, 882.3312988, -903.7961426, 972.5676880, -1790.9895020, 1786.1274414

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3014829, upper bound: 10002.3134902
time: 0.69 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3302955, upper bound: 10002.3341588
time: 0.75 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -6075.7607422, 6381.3359375, -5498.2578125, 5784.8613281, -11860.6220703, 11879.5927734
1: -642.4890747, 464.3061218, -582.2402344, 419.9122925, -1062.4012451, 1046.5462646
2: -1044.1705322, 1196.6938477, -945.6491699, 1086.3648682, -2130.5346680, 2142.3430176
3: -1192.0223389, 759.1086426, -1080.5843506, 687.4306641, -1879.4528809, 1839.6928711
4: -903.7961426, 972.5676880, -818.4217529, 882.3312988, -1786.1274414, 1790.9895020

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3134902, upper bound: 10002.3137316
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3341587, upper bound: 10002.3326672
time: 0.83 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -6075.7607422, 6381.3359375, -6075.7607422, 6381.3359375, -12457.0957031, 12457.0957031
1: -642.4890747, 464.3061218, -642.4890747, 464.3061218, -1106.7950439, 1106.7950439
2: -1044.1705322, 1196.6938477, -1044.1705322, 1196.6938477, -2240.8642578, 2240.8640137
3: -1192.0223389, 759.1086426, -1192.0223389, 759.1086426, -1951.1309814, 1951.1309814
4: -903.7961426, 972.5676880, -903.7961426, 972.5676880, -1876.3637695, 1876.3637695

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9999.0995924, upper bound: 10001.8601576
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9998.9509818, upper bound: 9998.9509818
time: 0.67 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.19 seconds
NS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 4.19
Output dim: 0, lower bound: -9999.0899976, upper bound: 10001.7685167
NS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 4.19
Output dim: 0, lower bound: -9999.0495044, upper bound: 10001.1447028
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 0, lower bound: -10002.3014829, upper bound: 10002.3134902
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 0, lower bound: -10002.3302955, upper bound: 10002.3341588
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 0, lower bound: -10002.3134902, upper bound: 10002.3137316
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 0, lower bound: -10002.3341587, upper bound: 10002.3326672
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 4.19
Output dim: 0, lower bound: -9999.0995924, upper bound: 10001.8601576
NS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 4.19
Output dim: 0, lower bound: -9998.9509818, upper bound: 9998.9509818

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -5447.0576172, 5732.4433594, -6194.5019531, 6520.5185547, -11967.5732422, 11926.9443359
1: -576.9442749, 416.0268555, -656.4984131, 472.9173279, -1049.8615723, 1072.5251465
2: -937.1323853, 1076.4998779, -1067.8941650, 1221.2786865, -2158.4111328, 2144.3937988
3: -1071.1759033, 681.1684570, -1220.1345215, 775.1709595, -1846.3465576, 1901.3028564
4: -811.2133179, 874.3626099, -924.4403076, 993.1643066, -1804.3775635, 1798.8028564

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3136920, upper bound: 10002.3133095
time: 0.70 seconds

## Relational analysis of NS_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3137315, upper bound: 10002.3134800
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -5498.2578125, 5784.8613281, -6042.5327148, 6347.2753906, -11845.5332031, 11827.3925781
1: -582.2402344, 419.9122925, -639.0449219, 461.7742920, -1044.0145264, 1058.9572754
2: -945.6491699, 1086.3648682, -1038.5175781, 1190.3339844, -2135.9831543, 2124.8815918
3: -1080.5843506, 687.4306641, -1185.6142578, 755.0041504, -1835.5883789, 1873.0447998
4: -818.4217529, 882.3312988, -898.9096069, 967.3844604, -1785.8061523, 1781.2407227

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_B2_B1

### Relational analysis result of NS_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.2507776, upper bound: 10001.3467187
time: 0.70 seconds

## Relational analysis of NS_A1_B2_B2_B2

### Relational analysis result of NS_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9999.0495044, upper bound: 10001.1447028
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -6194.5019531, 6520.5185547, -5447.0576172, 5732.4433594, -11926.9443359, 11967.5732422
1: -656.4984131, 472.9173279, -576.9442749, 416.0268555, -1072.5250244, 1049.8615723
2: -1067.8941650, 1221.2786865, -937.1323853, 1076.4998779, -2144.3937988, 2158.4111328
3: -1220.1345215, 775.1709595, -1071.1759033, 681.1684570, -1901.3028564, 1846.3465576
4: -924.4403076, 993.1643066, -811.2133179, 874.3626099, -1798.8028564, 1804.3775635

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3133095, upper bound: 10002.3136920
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3134800, upper bound: 10002.3137315
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -6042.5327148, 6347.2753906, -5498.2578125, 5784.8613281, -11827.3925781, 11845.5332031
1: -639.0449219, 461.7742920, -582.2402344, 419.9122925, -1058.9572754, 1044.0145264
2: -1038.5175781, 1190.3339844, -945.6491699, 1086.3648682, -2124.8815918, 2135.9831543
3: -1185.6142578, 755.0041504, -1080.5843506, 687.4306641, -1873.0447998, 1835.5883789
4: -898.9096069, 967.3844604, -818.4217529, 882.3312988, -1781.2407227, 1785.8061523

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10001.3467187, upper bound: 10002.2507776
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10001.1447028, upper bound: 9999.0495044
time: 0.70 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.29 seconds
NS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.29
Output dim: 0, lower bound: -10002.3136920, upper bound: 10002.3133095
NS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.29
Output dim: 0, lower bound: -10002.3137315, upper bound: 10002.3134800
NS_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.29
Output dim: 0, lower bound: -10002.2507776, upper bound: 10001.3467187
NS_A1_B2_B2_B2, status: Status.VERIFIED, split count: 4, time: 4.29
Output dim: 0, lower bound: -9999.0495044, upper bound: 10001.1447028
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.29
Output dim: 0, lower bound: -10002.3133095, upper bound: 10002.3136920
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.29
Output dim: 0, lower bound: -10002.3134800, upper bound: 10002.3137315
NS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.29
Output dim: 0, lower bound: -10001.3467187, upper bound: 10002.2507776
NS_A2_B1_A2_A2, status: Status.VERIFIED, split count: 4, time: 4.29
Output dim: 0, lower bound: -10001.1447028, upper bound: 9999.0495044

## BFS NS instance: NS_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -5088.1416016, 5366.0097656, -6026.2231445, 6353.3603516, -11441.5019531, 11392.2324219
1: -539.4620972, 388.6562195, -639.2252808, 460.0372009, -999.4992676, 1027.8814697
2: -875.6632690, 1006.3730469, -1039.1937256, 1188.8251953, -2064.4885254, 2045.5665283
3: -999.5240479, 636.6911621, -1186.4569092, 754.5945435, -1754.1186523, 1823.1480713
4: -757.4101562, 818.0061646, -898.9777222, 967.2737427, -1724.6837158, 1716.9836426

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_B1_A1_B1

### Relational analysis result of NS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3119804, upper bound: 10002.3090816
time: 0.70 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2

### Relational analysis result of NS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3095859, upper bound: 10002.3059296
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -5241.1938477, 5530.3002930, -6072.9404297, 6395.6459961, -11636.8398438, 11603.2402344
1: -555.8165894, 400.3746033, -643.6393433, 463.7269287, -1019.5434570, 1044.0137939
2: -900.2608032, 1038.3129883, -1046.4156494, 1197.7165527, -2097.9768066, 2084.7285156
3: -1027.5410156, 655.8336182, -1195.2875977, 760.0758057, -1787.6168213, 1851.1209717
4: -778.3610229, 843.6817017, -905.7948608, 974.1798096, -1752.5406494, 1749.4764404

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_B1_A2_B1

### Relational analysis result of NS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3090363, upper bound: 10002.3070735
time: 0.65 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2

### Relational analysis result of NS_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3090363, upper bound: 10002.3070735
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -5356.8881836, 5634.9785156, -5804.4218750, 6096.2270508, -11453.1132812, 11439.3994141
1: -567.2068481, 409.0914917, -614.0144043, 443.6669006, -1010.8736572, 1023.1058960
2: -921.4486084, 1058.7185059, -998.0986938, 1144.0753174, -2065.5239258, 2056.8171387
3: -1053.2047119, 669.6562500, -1140.6097412, 725.2817993, -1778.4864502, 1810.2659912
4: -797.6506958, 859.4206543, -864.7868652, 929.2726440, -1726.9229736, 1724.2073975

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B2_B1_A1

### Relational analysis result of NS_A1_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10002.0963160, upper bound: 10000.7635406
time: 0.71 seconds

## Relational analysis of NS_A1_B2_B2_B1_A2

### Relational analysis result of NS_A1_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10002.1294509, upper bound: 10000.5795007
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -6026.2231445, 6353.3603516, -5088.1416016, 5366.0097656, -11392.2324219, 11441.5019531
1: -639.2252808, 460.0372009, -539.4620972, 388.6562195, -1027.8814697, 999.4992676
2: -1039.1937256, 1188.8251953, -875.6632690, 1006.3730469, -2045.5666504, 2064.4885254
3: -1186.4569092, 754.5945435, -999.5240479, 636.6911621, -1823.1480713, 1754.1186523
4: -898.9777222, 967.2737427, -757.4101562, 818.0061646, -1716.9836426, 1724.6838379

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3090816, upper bound: 10002.3119804
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3059296, upper bound: 10002.3095859
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -6072.9404297, 6395.6459961, -5241.1938477, 5530.3002930, -11603.2402344, 11636.8398438
1: -643.6393433, 463.7269287, -555.8165894, 400.3746033, -1044.0139160, 1019.5434570
2: -1046.4156494, 1197.7165527, -900.2608032, 1038.3129883, -2084.7285156, 2097.9768066
3: -1195.2875977, 760.0758057, -1027.5410156, 655.8336182, -1851.1209717, 1787.6168213
4: -905.7948608, 974.1798096, -778.3610229, 843.6817017, -1749.4764404, 1752.5406494

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3070735, upper bound: 10002.3090363
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3070735, upper bound: 10002.3090363
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -5804.4218750, 6096.2270508, -5356.8881836, 5634.9785156, -11439.3994141, 11453.1132812
1: -614.0144043, 443.6669006, -567.2068481, 409.0914917, -1023.1058960, 1010.8736572
2: -998.0986938, 1144.0753174, -921.4486084, 1058.7185059, -2056.8171387, 2065.5239258
3: -1140.6097412, 725.2817993, -1053.2047119, 669.6562500, -1810.2659912, 1778.4864502
4: -864.7868652, 929.2726440, -797.6506958, 859.4206543, -1724.2073975, 1726.9229736

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_A1_B1

### Relational analysis result of NS_A2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10000.7635406, upper bound: 10002.0963160
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_A1_B2

### Relational analysis result of NS_A2_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10000.5795007, upper bound: 10002.1294509
time: 0.64 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.21 seconds
NS_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -10002.3119804, upper bound: 10002.3090816
NS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -10002.3095859, upper bound: 10002.3059296
NS_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -10002.3090363, upper bound: 10002.3070735
NS_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -10002.3090363, upper bound: 10002.3070735
NS_A1_B2_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.21
Output dim: 0, lower bound: -10002.0963160, upper bound: 10000.7635406
NS_A1_B2_B2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.21
Output dim: 0, lower bound: -10002.1294509, upper bound: 10000.5795007
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -10002.3090816, upper bound: 10002.3119804
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -10002.3059296, upper bound: 10002.3095859
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -10002.3070735, upper bound: 10002.3090363
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -10002.3070735, upper bound: 10002.3090363
NS_A2_B1_A2_A1_B1, status: Status.VERIFIED, split count: 5, time: 4.21
Output dim: 0, lower bound: -10000.7635406, upper bound: 10002.0963160
NS_A2_B1_A2_A1_B2, status: Status.VERIFIED, split count: 5, time: 4.21
Output dim: 0, lower bound: -10000.5795007, upper bound: 10002.1294509

## BFS NS instance: NS_A1_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -5059.2534180, 5337.3417969, -5965.3432617, 6293.2919922, -11352.5439453, 11302.6855469
1: -536.4877930, 386.4395752, -633.1748657, 455.4012146, -991.8890381, 1019.6143188
2: -870.7602539, 1001.0363159, -1029.0015869, 1177.5812988, -2048.3415527, 2030.0378418
3: -993.8976440, 633.1417236, -1174.9930420, 747.2328491, -1741.1304932, 1808.1347656
4: -753.0812378, 813.7339478, -890.1279907, 958.1772461, -1711.2584229, 1703.8616943

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -5070.1474609, 5348.5087891, -5984.6953125, 6312.9331055, -11383.0800781, 11333.2031250
1: -537.6541748, 387.2929993, -635.0447388, 456.8792419, -994.5334473, 1022.3377686
2: -872.6056519, 1003.1019897, -1032.1932373, 1181.1954346, -2053.8010254, 2035.2951660
3: -995.9919434, 634.5117188, -1178.4455566, 749.5462646, -1745.5380859, 1812.9572754
4: -754.6970825, 815.3722534, -892.8028564, 961.1187134, -1715.8155518, 1708.1749268

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -5227.7275391, 5517.7709961, -5942.0732422, 6268.3227539, -11496.0507812, 11459.8437500
1: -554.4888306, 399.3483887, -630.4281616, 453.7631226, -1008.2519531, 1029.7762451
2: -897.9366455, 1035.8508301, -1023.9167480, 1173.2492676, -2071.1857910, 2059.7675781
3: -1024.7386475, 654.2616577, -1168.7590332, 744.4288940, -1769.1673584, 1823.0207520
4: -776.2213745, 841.7360840, -885.5362549, 954.5786133, -1730.7999268, 1727.2723389

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3090363, upper bound: 10002.3070735
time: 0.88 seconds

## Relational analysis of NS_A1_B2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3090363, upper bound: 10002.3070735
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -5195.3564453, 5483.6245117, -7917.5698242, 8510.7177734, -13706.0742188, 13401.1933594
1: -551.0971069, 396.8045349, -851.7946167, 602.8843384, -1153.9810791, 1248.5991211
2: -892.4678345, 1029.5150146, -1363.7227783, 1584.5545654, -2477.0224609, 2393.2370605
3: -1018.7199097, 650.1846313, -1545.4357910, 1004.5301514, -2023.2500000, 2195.6203613
4: -771.6190796, 836.5921021, -1169.2437744, 1290.4058838, -2062.0246582, 2005.8358154

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3090363, upper bound: 10002.3070735
time: 0.67 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3090363, upper bound: 10002.3070735
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -5965.3432617, 6293.2919922, -5059.2534180, 5337.3417969, -11302.6855469, 11352.5449219
1: -633.1748657, 455.4012146, -536.4877930, 386.4395752, -1019.6143188, 991.8890381
2: -1029.0015869, 1177.5812988, -870.7602539, 1001.0363159, -2030.0378418, 2048.3415527
3: -1174.9930420, 747.2328491, -993.8976440, 633.1417236, -1808.1347656, 1741.1304932
4: -890.1279907, 958.1772461, -753.0812378, 813.7339478, -1703.8616943, 1711.2585449

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -5984.6953125, 6312.9331055, -5070.1474609, 5348.5087891, -11333.2041016, 11383.0800781
1: -635.0447388, 456.8792419, -537.6541748, 387.2929993, -1022.3377686, 994.5334473
2: -1032.1932373, 1181.1954346, -872.6056519, 1003.1019897, -2035.2951660, 2053.8010254
3: -1178.4455566, 749.5462646, -995.9919434, 634.5117188, -1812.9572754, 1745.5380859
4: -892.8028564, 961.1187134, -754.6970825, 815.3722534, -1708.1749268, 1715.8155518

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -5942.0732422, 6268.3227539, -5227.7275391, 5517.7709961, -11459.8437500, 11496.0507812
1: -630.4281616, 453.7631226, -554.4888306, 399.3483887, -1029.7763672, 1008.2519531
2: -1023.9167480, 1173.2492676, -897.9366455, 1035.8508301, -2059.7675781, 2071.1857910
3: -1168.7590332, 744.4288940, -1024.7386475, 654.2616577, -1823.0207520, 1769.1673584
4: -885.5362549, 954.5786133, -776.2213745, 841.7360840, -1727.2723389, 1730.7999268

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3070735, upper bound: 10002.3090363
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3070735, upper bound: 10002.3090363
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7917.5698242, 8510.7177734, -5195.3564453, 5483.6245117, -13401.1933594, 13706.0742188
1: -851.7946167, 602.8843384, -551.0971069, 396.8045349, -1248.5991211, 1153.9810791
2: -1363.7227783, 1584.5545654, -892.4678345, 1029.5150146, -2393.2373047, 2477.0224609
3: -1545.4357910, 1004.5301514, -1018.7199097, 650.1846313, -2195.6203613, 2023.2500000
4: -1169.2437744, 1290.4058838, -771.6190796, 836.5921021, -2005.8358154, 2062.0244141

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3070735, upper bound: 10002.3090363
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3070735, upper bound: 10002.3090363
time: 0.67 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 5.88 seconds
NS_A1_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 0, lower bound: -10002.3090363, upper bound: 10002.3070735
NS_A1_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 0, lower bound: -10002.3090363, upper bound: 10002.3070735
NS_A1_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 0, lower bound: -10002.3090363, upper bound: 10002.3070735
NS_A1_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 0, lower bound: -10002.3090363, upper bound: 10002.3070735
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 0, lower bound: -10002.3070735, upper bound: 10002.3090363
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 0, lower bound: -10002.3070735, upper bound: 10002.3090363
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 0, lower bound: -10002.3070735, upper bound: 10002.3090363
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 0, lower bound: -10002.3070735, upper bound: 10002.3090363

## BFS NS instance: NS_A1_B2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5126.9096680, 5423.8027344, -5942.0732422, 6268.3227539, -11395.2324219, 11365.8759766
1: -544.5504761, 391.6739197, -630.4281616, 453.7631226, -998.3135986, 1022.1020508
2: -880.5272217, 1017.4307861, -1023.9167480, 1173.2492676, -2053.7758789, 2041.3475342
3: -1003.7624512, 642.4940186, -1168.7590332, 744.4288940, -1748.1910400, 1811.2528076
4: -760.2072144, 827.1680298, -885.5362549, 954.5786133, -1714.7857666, 1712.7043457

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_B1_A2_B1_A1_A1

### Relational analysis result of NS_A1_B2_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3036423, upper bound: 10002.2957136
time: 0.69 seconds

## Relational analysis of NS_A1_B2_B1_A2_B1_A1_A2

### Relational analysis result of NS_A1_B2_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3036423, upper bound: 10002.3070735
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -6865.1328125, 7352.3784180, -5942.0732422, 6268.3227539, -13133.4550781, 13294.4511719
1: -736.4389038, 522.9152832, -630.4281616, 453.7631226, -1190.2019043, 1153.3431396
2: -1180.4207764, 1369.8156738, -1023.9167480, 1173.2492676, -2353.6699219, 2393.7321777
3: -1338.4526367, 868.3823242, -1168.7590332, 744.4288940, -2082.8815918, 2037.1413574
4: -1012.6337891, 1116.0017090, -885.5362549, 954.5786133, -1967.2124023, 2001.5379639

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_B1_A2_B1_A2_A1

### Relational analysis result of NS_A1_B2_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3036423, upper bound: 10002.2957136
time: 0.77 seconds

## Relational analysis of NS_A1_B2_B1_A2_B1_A2_A2

### Relational analysis result of NS_A1_B2_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3036423, upper bound: 10002.3070735
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5126.9096680, 5423.8027344, -7917.5698242, 8510.7177734, -13637.6269531, 13341.3710938
1: -544.5504761, 391.6739197, -851.7946167, 602.8843384, -1147.4346924, 1243.4685059
2: -880.5272217, 1017.4307861, -1363.7227783, 1584.5545654, -2465.0817871, 2381.1530762
3: -1003.7624512, 642.4940186, -1545.4357910, 1004.5301514, -2008.2923584, 2187.9296875
4: -760.2072144, 827.1680298, -1169.2437744, 1290.4058838, -2050.6130371, 1996.4118652

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_A1

### Relational analysis result of NS_A1_B2_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3036413, upper bound: 10002.2957136
time: 0.57 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_A2

### Relational analysis result of NS_A1_B2_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3036413, upper bound: 10002.3070735
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6865.1328125, 7352.3784180, -7917.5698242, 8510.7177734, -15375.8505859, 15269.9472656
1: -736.4389038, 522.9152832, -851.7946167, 602.8843384, -1339.3231201, 1374.7098389
2: -1180.4207764, 1369.8156738, -1363.7227783, 1584.5545654, -2764.9753418, 2733.5380859
3: -1338.4526367, 868.3823242, -1545.4357910, 1004.5301514, -2342.9829102, 2413.8181152
4: -1012.6337891, 1116.0017090, -1169.2437744, 1290.4058838, -2303.0395508, 2285.2456055

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_A1

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3036413, upper bound: 10002.2957136
time: 0.67 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_A2

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3036413, upper bound: 10002.3070735
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -5942.0732422, 6268.3227539, -5126.9096680, 5423.8027344, -11365.8759766, 11395.2324219
1: -630.4281616, 453.7631226, -544.5504761, 391.6739197, -1022.1020508, 998.3135986
2: -1023.9167480, 1173.2492676, -880.5272217, 1017.4307861, -2041.3475342, 2053.7761230
3: -1168.7590332, 744.4288940, -1003.7624512, 642.4940186, -1811.2528076, 1748.1910400
4: -885.5362549, 954.5786133, -760.2072144, 827.1680298, -1712.7043457, 1714.7858887

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.2957136, upper bound: 10002.3036423
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.2957136, upper bound: 10002.3036423
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -5942.0732422, 6268.3227539, -6865.1328125, 7352.3784180, -13294.4511719, 13133.4550781
1: -630.4281616, 453.7631226, -736.4389038, 522.9152832, -1153.3431396, 1190.2019043
2: -1023.9167480, 1173.2492676, -1180.4207764, 1369.8156738, -2393.7324219, 2353.6699219
3: -1168.7590332, 744.4288940, -1338.4526367, 868.3823242, -2037.1413574, 2082.8815918
4: -885.5362549, 954.5786133, -1012.6337891, 1116.0017090, -2001.5379639, 1967.2122803

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.2957136, upper bound: 10002.3036423
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.2957136, upper bound: 10002.3090363
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -7917.5698242, 8510.7177734, -5126.9096680, 5423.8027344, -13341.3710938, 13637.6269531
1: -851.7946167, 602.8843384, -544.5504761, 391.6739197, -1243.4685059, 1147.4346924
2: -1363.7227783, 1584.5545654, -880.5272217, 1017.4307861, -2381.1530762, 2465.0817871
3: -1545.4357910, 1004.5301514, -1003.7624512, 642.4940186, -2187.9296875, 2008.2923584
4: -1169.2437744, 1290.4058838, -760.2072144, 827.1680298, -1996.4118652, 2050.6130371

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.2991535, upper bound: 10002.2756226
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3070735, upper bound: 10002.3084741
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -7917.5698242, 8510.7177734, -6865.1328125, 7352.3784180, -15269.9472656, 15375.8505859
1: -851.7946167, 602.8843384, -736.4389038, 522.9152832, -1374.7098389, 1339.3231201
2: -1363.7227783, 1584.5545654, -1180.4207764, 1369.8156738, -2733.5380859, 2764.9753418
3: -1545.4357910, 1004.5301514, -1338.4526367, 868.3823242, -2413.8181152, 2342.9829102
4: -1169.2437744, 1290.4058838, -1012.6337891, 1116.0017090, -2285.2453613, 2303.0395508

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3049357, upper bound: 10002.2978506
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3070735, upper bound: 10002.3084741
time: 0.66 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 5.48 seconds
NS_A1_B2_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 5.48
Output dim: 0, lower bound: -10002.3036423, upper bound: 10002.2957136
NS_A1_B2_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 5.48
Output dim: 0, lower bound: -10002.3036423, upper bound: 10002.3070735
NS_A1_B2_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.48
Output dim: 0, lower bound: -10002.3036423, upper bound: 10002.2957136
NS_A1_B2_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.48
Output dim: 0, lower bound: -10002.3036423, upper bound: 10002.3070735
NS_A1_B2_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 5.48
Output dim: 0, lower bound: -10002.3036413, upper bound: 10002.2957136
NS_A1_B2_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 5.48
Output dim: 0, lower bound: -10002.3036413, upper bound: 10002.3070735
NS_A1_B2_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.48
Output dim: 0, lower bound: -10002.3036413, upper bound: 10002.2957136
NS_A1_B2_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.48
Output dim: 0, lower bound: -10002.3036413, upper bound: 10002.3070735
NS_A2_B1_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.48
Output dim: 0, lower bound: -10002.2957136, upper bound: 10002.3036423
NS_A2_B1_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.48
Output dim: 0, lower bound: -10002.2957136, upper bound: 10002.3036423
NS_A2_B1_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.48
Output dim: 0, lower bound: -10002.2957136, upper bound: 10002.3036423
NS_A2_B1_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.48
Output dim: 0, lower bound: -10002.2957136, upper bound: 10002.3090363
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.48
Output dim: 0, lower bound: -10002.2991535, upper bound: 10002.2756226
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.48
Output dim: 0, lower bound: -10002.3070735, upper bound: 10002.3084741
NS_A2_B1_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.48
Output dim: 0, lower bound: -10002.3049357, upper bound: 10002.2978506
NS_A2_B1_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.48
Output dim: 0, lower bound: -10002.3070735, upper bound: 10002.3084741

## BFS NS instance: NS_A1_B2_B1_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -5302.9794922, 5603.2553711, -5942.0732422, 6268.3227539, -11571.3027344, 11545.3281250
1: -563.5404663, 404.4458618, -630.4281616, 453.7631226, -1017.3035889, 1034.8739014
2: -915.0772705, 1049.8432617, -1023.9167480, 1173.2492676, -2088.3264160, 2073.7600098
3: -1046.5393066, 665.0908813, -1168.7590332, 744.4288940, -1790.9680176, 1833.8498535
4: -791.9868774, 854.4829102, -885.5362549, 954.5786133, -1746.5654297, 1740.0191650

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 39

## BFS NS instance: NS_A1_B2_B1_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -5142.8994141, 5440.7504883, -5942.0732422, 6268.3227539, -11411.2226562, 11382.8242188
1: -546.2619629, 392.8964539, -630.4281616, 453.7631226, -1000.0250854, 1023.3245850
2: -883.1278076, 1020.6369019, -1023.9167480, 1173.2492676, -2056.3767090, 2044.5537109
3: -1006.5407715, 644.4893799, -1168.7590332, 744.4288940, -1750.9694824, 1813.2484131
4: -762.3411255, 829.7717285, -885.5362549, 954.5786133, -1716.9196777, 1715.3077393

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 11

## BFS NS instance: NS_A1_B2_B1_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -7311.5146484, 7876.8349609, -5942.0732422, 6268.3227539, -13579.8378906, 13818.9082031
1: -787.9283447, 556.1011963, -630.4281616, 453.7631226, -1241.6914062, 1186.5291748
2: -1260.0800781, 1465.6712646, -1023.9167480, 1173.2492676, -2433.3288574, 2489.5878906
3: -1429.0966797, 928.9824829, -1168.7590332, 744.4288940, -2173.5256348, 2097.7414551
4: -1080.5386963, 1194.6652832, -885.5362549, 954.5786133, -2035.1173096, 2080.2016602

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A1_B2_B1_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -6881.6542969, 7368.0859375, -5942.0732422, 6268.3227539, -13149.9765625, 13310.1591797
1: -738.0714722, 524.1710205, -630.4281616, 453.7631226, -1191.8343506, 1154.5988770
2: -1183.0433350, 1372.7709961, -1023.9167480, 1173.2492676, -2356.2924805, 2396.6875000
3: -1341.4683838, 870.2890625, -1168.7590332, 744.4288940, -2085.8972168, 2039.0480957
4: -1014.9092407, 1118.3824463, -885.5362549, 954.5786133, -1969.4877930, 2003.9185791

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A1_B2_B1_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -5302.9794922, 5603.2553711, -7917.5698242, 8510.7177734, -13813.6953125, 13520.8251953
1: -563.5404663, 404.4458618, -851.7946167, 602.8843384, -1166.4244385, 1256.2404785
2: -915.0772705, 1049.8432617, -1363.7227783, 1584.5545654, -2499.6318359, 2413.5651855
3: -1046.5393066, 665.0908813, -1545.4357910, 1004.5301514, -2051.0693359, 2210.5266113
4: -791.9868774, 854.4829102, -1169.2437744, 1290.4058838, -2082.3928223, 2023.7266846

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 12

## BFS NS instance: NS_A1_B2_B1_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -5142.8994141, 5440.7504883, -7917.5698242, 8510.7177734, -13653.6171875, 13358.3193359
1: -546.2619629, 392.8964539, -851.7946167, 602.8843384, -1149.1462402, 1244.6910400
2: -883.1278076, 1020.6369019, -1363.7227783, 1584.5545654, -2467.6823730, 2384.3591309
3: -1006.5407715, 644.4893799, -1545.4357910, 1004.5301514, -2011.0708008, 2189.9252930
4: -762.3411255, 829.7717285, -1169.2437744, 1290.4058838, -2052.7468262, 1999.0151367

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 12

## BFS NS instance: NS_A1_B2_B1_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -7311.5146484, 7876.8349609, -7917.5698242, 8510.7177734, -15822.2324219, 15794.4042969
1: -787.9283447, 556.1011963, -851.7946167, 602.8843384, -1390.8127441, 1407.8957520
2: -1260.0800781, 1465.6712646, -1363.7227783, 1584.5545654, -2844.6342773, 2829.3940430
3: -1429.0966797, 928.9824829, -1545.4357910, 1004.5301514, -2433.6267090, 2474.4182129
4: -1080.5386963, 1194.6652832, -1169.2437744, 1290.4058838, -2370.9443359, 2363.9091797

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 39

## BFS NS instance: NS_A1_B2_B1_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -6881.6542969, 7368.0859375, -7917.5698242, 8510.7177734, -15392.3720703, 15285.6542969
1: -738.0714722, 524.1710205, -851.7946167, 602.8843384, -1340.9556885, 1375.9655762
2: -1183.0433350, 1372.7709961, -1363.7227783, 1584.5545654, -2767.5979004, 2736.4936523
3: -1341.4683838, 870.2890625, -1545.4357910, 1004.5301514, -2345.9985352, 2415.7248535
4: -1014.9092407, 1118.3824463, -1169.2437744, 1290.4058838, -2305.3151855, 2287.6257324

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 39

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -5942.0732422, 6268.3227539, -5302.9794922, 5603.2553711, -11545.3281250, 11571.3027344
1: -630.4281616, 453.7631226, -563.5404663, 404.4458618, -1034.8739014, 1017.3035889
2: -1023.9167480, 1173.2492676, -915.0772705, 1049.8432617, -2073.7600098, 2088.3264160
3: -1168.7590332, 744.4288940, -1046.5393066, 665.0908813, -1833.8498535, 1790.9680176
4: -885.5362549, 954.5786133, -791.9868774, 854.4829102, -1740.0191650, 1746.5654297

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -5942.0732422, 6268.3227539, -5142.8994141, 5440.7504883, -11382.8242188, 11411.2226562
1: -630.4281616, 453.7631226, -546.2619629, 392.8964539, -1023.3245850, 1000.0250854
2: -1023.9167480, 1173.2492676, -883.1278076, 1020.6369019, -2044.5537109, 2056.3767090
3: -1168.7590332, 744.4288940, -1006.5407715, 644.4893799, -1813.2484131, 1750.9694824
4: -885.5362549, 954.5786133, -762.3411255, 829.7717285, -1715.3077393, 1716.9196777

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 11

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -5942.0732422, 6268.3227539, -7311.5146484, 7876.8349609, -13818.9082031, 13579.8378906
1: -630.4281616, 453.7631226, -787.9283447, 556.1011963, -1186.5290527, 1241.6914062
2: -1023.9167480, 1173.2492676, -1260.0800781, 1465.6712646, -2489.5878906, 2433.3288574
3: -1168.7590332, 744.4288940, -1429.0966797, 928.9824829, -2097.7414551, 2173.5256348
4: -885.5362549, 954.5786133, -1080.5386963, 1194.6652832, -2080.2016602, 2035.1173096

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 12

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -5942.0732422, 6268.3227539, -6881.6542969, 7368.0859375, -13310.1591797, 13149.9765625
1: -630.4281616, 453.7631226, -738.0714722, 524.1710205, -1154.5988770, 1191.8343506
2: -1023.9167480, 1173.2492676, -1183.0433350, 1372.7709961, -2396.6877441, 2356.2924805
3: -1168.7590332, 744.4288940, -1341.4683838, 870.2890625, -2039.0480957, 2085.8972168
4: -885.5362549, 954.5786133, -1014.9092407, 1118.3824463, -2003.9185791, 1969.4877930

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 12

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7793.1176758, 8374.9609375, -5077.9570312, 5373.7763672, -13166.8935547, 13452.9179688
1: -838.3736572, 593.3051147, -539.4947510, 387.9195251, -1226.2932129, 1132.7998047
2: -1342.2769775, 1559.3302002, -871.9857788, 1007.9796143, -2350.2565918, 2431.3159180
3: -1521.6136475, 988.5606079, -993.8884888, 636.4585571, -2158.0720215, 1982.4490967
4: -1151.1014404, 1269.7745361, -752.7149658, 819.4912720, -1970.5927734, 2022.4893799

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.2994594, upper bound: 10002.2759111
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.2994593, upper bound: 10002.2759111
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7796.9462891, 8377.7597656, -5052.4252930, 5343.7797852, -13140.7255859, 13430.1845703
1: -838.7290039, 593.6675415, -536.7026367, 386.0802002, -1224.8092041, 1130.3701172
2: -1343.2802734, 1559.9318848, -867.6237183, 1002.8067627, -2346.0866699, 2427.5556641
3: -1523.0452881, 989.0667114, -989.5325317, 633.1461182, -2156.1914062, 1978.5992432
4: -1152.3369141, 1270.1206055, -749.4940186, 814.9962158, -1967.3331299, 2019.6146240

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3058778, upper bound: 10002.2978506
time: 0.89 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.3058778, upper bound: 10002.3134919
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -7846.1337891, 8432.9218750, -6742.8750000, 7216.7812500, -15062.9150391, 15175.7958984
1: -844.0782471, 597.4169922, -723.0516357, 513.4302979, -1357.5085449, 1320.4682617
2: -1351.3765869, 1570.1257324, -1159.5770264, 1344.6026611, -2695.9792480, 2729.7026367
3: -1531.6032715, 995.3789062, -1315.4476318, 852.4293823, -2384.0327148, 2310.8264160
4: -1158.7476807, 1278.5844727, -995.1806641, 1095.3635254, -2254.1110840, 2273.7651367

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.2991535, upper bound: 10002.2756226
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.2991535, upper bound: 10002.2978506
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -7842.1616211, 8427.5771484, -6750.2167969, 7224.8637695, -15067.0244141, 15177.7929688
1: -843.6242065, 597.1218872, -723.9249268, 514.0421753, -1357.6661377, 1321.0468750
2: -1350.9399414, 1569.1604004, -1161.1328125, 1346.1708984, -2697.1108398, 2730.2932129
3: -1531.4320068, 994.8596802, -1317.5010986, 853.5619507, -2384.9938965, 2312.3608398
4: -1158.6676025, 1277.7239990, -996.8314209, 1096.5871582, -2255.2543945, 2274.5551758

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.2991535, upper bound: 10002.2756226
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10002.2991535, upper bound: 10002.3084741
time: 0.75 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 6.00 seconds
NS_A2_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.00
Output dim: 0, lower bound: -10002.2994594, upper bound: 10002.2759111
NS_A2_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.00
Output dim: 0, lower bound: -10002.2994593, upper bound: 10002.2759111
NS_A2_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.00
Output dim: 0, lower bound: -10002.3058778, upper bound: 10002.2978506
NS_A2_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.00
Output dim: 0, lower bound: -10002.3058778, upper bound: 10002.3134919
NS_A2_B1_A1_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 6.00
Output dim: 0, lower bound: -10002.2991535, upper bound: 10002.2756226
NS_A2_B1_A1_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 6.00
Output dim: 0, lower bound: -10002.2991535, upper bound: 10002.2978506
NS_A2_B1_A1_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 6.00
Output dim: 0, lower bound: -10002.2991535, upper bound: 10002.2756226
NS_A2_B1_A1_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 6.00
Output dim: 0, lower bound: -10002.2991535, upper bound: 10002.3084741

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -7793.1176758, 8374.9609375, -5036.3325195, 5330.4428711, -13123.5605469, 13411.2929688
1: -838.3736572, 593.3051147, -535.2426147, 384.7048645, -1223.0784912, 1128.5477295
2: -1342.2769775, 1559.3302002, -864.2172852, 1000.0108643, -2342.2878418, 2423.5471191
3: -1521.6136475, 988.5606079, -984.9057007, 631.2650757, -2152.8779297, 1973.4663086
4: -1151.1014404, 1269.7745361, -745.7632446, 812.9667969, -1964.0682373, 2015.5375977

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -7793.1176758, 8374.9609375, -5009.1762695, 5297.4838867, -13090.6005859, 13384.1367188
1: -838.3736572, 593.3051147, -532.1813354, 382.8404541, -1221.2141113, 1125.4864502
2: -1342.2769775, 1559.3302002, -860.1756592, 994.3784180, -2336.6552734, 2419.5056152
3: -1521.6136475, 988.5606079, -981.4010010, 627.7311401, -2149.3442383, 1969.9614258
4: -1151.1014404, 1269.7745361, -743.3620605, 807.9497681, -1959.0511475, 2013.1364746

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7796.9462891, 8377.7597656, -5036.3325195, 5330.4428711, -13127.3886719, 13414.0917969
1: -838.7290039, 593.6675415, -535.2426147, 384.7048645, -1223.4337158, 1128.9101562
2: -1343.2802734, 1559.9318848, -864.2172852, 1000.0108643, -2343.2910156, 2424.1491699
3: -1523.0452881, 989.0667114, -984.9057007, 631.2650757, -2154.3103027, 1973.9724121
4: -1152.3369141, 1270.1206055, -745.7632446, 812.9667969, -1965.3037109, 2015.8837891

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7796.9462891, 8377.7597656, -5009.1762695, 5297.4838867, -13094.4287109, 13386.9355469
1: -838.7290039, 593.6675415, -532.1813354, 382.8404541, -1221.5694580, 1125.8488770
2: -1343.2802734, 1559.9318848, -860.1756592, 994.3784180, -2337.6586914, 2420.1074219
3: -1523.0452881, 989.0667114, -981.4010010, 627.7311401, -2150.7763672, 1970.4676514
4: -1152.3369141, 1270.1206055, -743.3620605, 807.9497681, -1960.2864990, 2013.4826660

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -7793.1176758, 8374.9609375, -6742.8750000, 7216.7812500, -15009.8984375, 15117.8330078
1: -838.3736572, 593.3051147, -723.0516357, 513.4302979, -1351.8039551, 1316.3566895
2: -1342.2769775, 1559.3302002, -1159.5770264, 1344.6026611, -2686.8796387, 2718.9072266
3: -1521.6136475, 988.5606079, -1315.4476318, 852.4293823, -2374.0429688, 2304.0083008
4: -1151.1014404, 1269.7745361, -995.1806641, 1095.3635254, -2246.4648438, 2264.9545898

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -7796.9462891, 8377.7597656, -6742.8750000, 7216.7812500, -15013.7275391, 15120.6328125
1: -838.7290039, 593.6675415, -723.0516357, 513.4302979, -1352.1593018, 1316.7191162
2: -1343.2802734, 1559.9318848, -1159.5770264, 1344.6026611, -2687.8828125, 2719.5087891
3: -1523.0452881, 989.0667114, -1315.4476318, 852.4293823, -2375.4746094, 2304.5141602
4: -1152.3369141, 1270.1206055, -995.1806641, 1095.3635254, -2247.6999512, 2265.3010254

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -7793.1176758, 8374.9609375, -6750.2167969, 7224.8637695, -15017.9804688, 15125.1767578
1: -838.3736572, 593.3051147, -723.9249268, 514.0421753, -1352.4156494, 1317.2299805
2: -1342.2769775, 1559.3302002, -1161.1328125, 1346.1708984, -2688.4477539, 2720.4624023
3: -1521.6136475, 988.5606079, -1317.5010986, 853.5619507, -2375.1755371, 2306.0617676
4: -1151.1014404, 1269.7745361, -996.8314209, 1096.5871582, -2247.6884766, 2266.6059570

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -7796.9462891, 8377.7597656, -6750.2167969, 7224.8637695, -15021.8095703, 15127.9765625
1: -838.7290039, 593.6675415, -723.9249268, 514.0421753, -1352.7709961, 1317.5925293
2: -1343.2802734, 1559.9318848, -1161.1328125, 1346.1708984, -2689.4511719, 2721.0646973
3: -1523.0452881, 989.0667114, -1317.5010986, 853.5619507, -2376.6071777, 2306.5678711
4: -1152.3369141, 1270.1206055, -996.8314209, 1096.5871582, -2248.9233398, 2266.9521484

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.81 + 281.13 = 285.94 seconds
