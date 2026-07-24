## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 71.14967792064


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-86.8210449, 164.3859558, -86.8210449, 164.3859558, -251.2070007, 251.2070007)
1: (-29.7035065, 57.3727303, -29.7035065, 57.3727303, -87.0762329, 87.0762329)
2: (-15.8138723, 59.4475250, -15.8138723, 59.4475250, -75.2613983, 75.2613983)
3: (-33.6637955, 71.3761292, -33.6637955, 71.3761292, -105.0399246, 105.0399246)
4: (-20.2065468, 58.7098618, -20.2065468, 58.7098618, -78.9164124, 78.9164124)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.06 + 1.92 = 3.98 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -71.1567936, upper bound: 71.1567936

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1552837, upper bound: 71.1544852
time: 4.40 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1552837, upper bound: 71.1555175
time: 0.70 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 5.28 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 5.28
Output dim: 4, lower bound: -71.1552837, upper bound: 71.1544852
NS_A2, status: Status.UNKNOWN, split count: 1, time: 5.28
Output dim: 4, lower bound: -71.1552837, upper bound: 71.1555175

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -76.9273148, 145.0817108, -86.8210449, 164.3859558, -241.3132477, 231.9027557
1: -26.1313763, 50.2405090, -29.7035065, 57.3727303, -83.5041046, 79.9440002
2: -13.9303913, 52.0849991, -15.8138723, 59.4475250, -73.3779144, 67.8988724
3: -29.6245155, 62.4448929, -33.6637955, 71.3761292, -101.0006409, 96.1086884
4: -17.7766476, 51.4184837, -20.2065468, 58.7098618, -76.4865036, 71.6250305

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1542386, upper bound: 71.1542386
time: 0.66 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1542386, upper bound: 71.1542386
time: 0.70 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -91.9336090, 174.0288239, -86.8210449, 164.3859558, -256.3195801, 260.8498535
1: -31.5269642, 60.6732368, -29.7035065, 57.3727303, -88.8996964, 90.3767395
2: -16.7808247, 62.8047867, -15.8138723, 59.4475250, -76.2283478, 78.6186600
3: -35.6499596, 75.4741058, -33.6637955, 71.3761292, -107.0260925, 109.1379013
4: -21.4301624, 62.0332565, -20.2065468, 58.7098618, -80.1400223, 82.2397995

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1544852, upper bound: 71.1552837
time: 0.69 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1544852, upper bound: 71.1555175
time: 0.73 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.53 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.53
Output dim: 4, lower bound: -71.1542386, upper bound: 71.1542386
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.53
Output dim: 4, lower bound: -71.1542386, upper bound: 71.1542386
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.53
Output dim: 4, lower bound: -71.1544852, upper bound: 71.1552837
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.53
Output dim: 4, lower bound: -71.1544852, upper bound: 71.1555175

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -76.9273148, 145.0817108, -76.9273148, 145.0817108, -222.0090027, 222.0090027
1: -26.1313763, 50.2405090, -26.1313763, 50.2405090, -76.3718796, 76.3718796
2: -13.9303913, 52.0849991, -13.9303913, 52.0849991, -66.0153885, 66.0153885
3: -29.6245155, 62.4448929, -29.6245155, 62.4448929, -92.0693970, 92.0693970
4: -17.7766476, 51.4184837, -17.7766476, 51.4184837, -69.1951294, 69.1951294

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1481925, upper bound: 71.1539740
time: 1.23 seconds

## Relational analysis of NS_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1538509, upper bound: 71.1538509
time: 0.90 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -76.9273148, 145.0817108, -91.9336090, 174.0288239, -250.9561462, 237.0153198
1: -26.1313763, 50.2405090, -31.5269642, 60.6732368, -86.8046112, 81.7674637
2: -13.9303913, 52.0849991, -16.7808247, 62.8047867, -76.7351761, 68.8658218
3: -29.6245155, 62.4448929, -35.6499596, 75.4741058, -105.0986176, 98.0948486
4: -17.7766476, 51.4184837, -21.4301624, 62.0332565, -79.8098907, 72.8486481

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1539740, upper bound: 71.1484101
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1538509, upper bound: 71.1540665
time: 0.64 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -91.9336090, 174.0288239, -76.9273148, 145.0817108, -237.0153198, 250.9561005
1: -31.5269642, 60.6732368, -26.1313763, 50.2405090, -81.7674637, 86.8046112
2: -16.7808247, 62.8047867, -13.9303913, 52.0849991, -68.8658218, 76.7351761
3: -35.6499596, 75.4741058, -29.6245155, 62.4448929, -98.0948486, 105.0986176
4: -21.4301624, 62.0332565, -17.7766476, 51.4184837, -72.8486481, 79.8098907

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1484101, upper bound: 71.1550321
time: 0.69 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1484101, upper bound: 71.1549555
time: 0.76 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -91.9336090, 174.0288239, -91.9336090, 174.0288239, -265.9624329, 265.9624329
1: -31.5269642, 60.6732368, -31.5269642, 60.6732368, -92.2002029, 92.2002029
2: -16.7808247, 62.8047867, -16.7808247, 62.8047867, -79.5856094, 79.5856094
3: -35.6499596, 75.4741058, -35.6499596, 75.4741058, -111.1240692, 111.1240692
4: -21.4301624, 62.0332565, -21.4301624, 62.0332565, -83.4634171, 83.4634171

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1542141, upper bound: 71.1501590
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1540665, upper bound: 71.1551732
time: 0.65 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.47 seconds
NS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 4, lower bound: -71.1481925, upper bound: 71.1539740
NS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 4, lower bound: -71.1538509, upper bound: 71.1538509
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 4, lower bound: -71.1539740, upper bound: 71.1484101
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 4, lower bound: -71.1538509, upper bound: 71.1540665
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 4, lower bound: -71.1484101, upper bound: 71.1550321
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 4, lower bound: -71.1484101, upper bound: 71.1549555
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 4, lower bound: -71.1542141, upper bound: 71.1501590
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 4, lower bound: -71.1540665, upper bound: 71.1551732

## BFS NS instance: NS_A1_B1_B1

### Backsubstitution after applying NS history:
0: -76.1778641, 143.5461731, -65.8488083, 123.1608505, -199.3386993, 209.3949890
1: -25.8639164, 49.7324600, -22.2078342, 42.6931381, -68.5570450, 71.9402618
2: -13.7920008, 51.5450745, -11.8839025, 44.2537460, -58.0457458, 63.4289742
3: -29.3275833, 61.8032494, -25.3091488, 53.0345116, -82.3620911, 87.1123962
4: -17.6015034, 50.8785400, -15.1839027, 43.6456490, -61.2471504, 66.0624390

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1481925, upper bound: 71.1481925
time: 0.68 seconds

## Relational analysis of NS_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1481925, upper bound: 71.1538509
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_B2

### Backsubstitution after applying NS history:
0: -76.9273148, 145.0817108, -73.7855072, 139.1022339, -216.0295105, 218.8672180
1: -26.1313763, 50.2405090, -25.0173073, 48.1168098, -74.2481766, 75.2578049
2: -13.9303913, 52.0849991, -13.3542852, 49.9176254, -63.8480148, 65.4392853
3: -29.6245155, 62.4448929, -28.4059334, 59.8230743, -89.4475861, 90.8508224
4: -17.7766476, 51.4184837, -17.0431519, 49.2884712, -67.0651169, 68.4616394

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1538509, upper bound: 71.1481925
time: 0.68 seconds

## Relational analysis of NS_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1538509, upper bound: 71.1538509
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -65.8488083, 123.1608505, -91.2265167, 172.5582886, -238.4071045, 214.3873444
1: -22.2078342, 42.6931381, -31.2533226, 60.1206589, -82.3284912, 73.9464569
2: -11.8839025, 44.2537460, -16.6439972, 62.2332649, -74.1171570, 60.8977432
3: -25.3091488, 53.0345116, -35.3476410, 74.7859116, -100.0950623, 88.3821564
4: -15.1839027, 43.6456490, -21.2512856, 61.4800644, -76.6639557, 64.8969269

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1481925, upper bound: 71.1484101
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1481925, upper bound: 71.1484101
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -73.7855072, 139.1022339, -91.9336090, 174.0288239, -247.8143311, 231.0358276
1: -25.0173073, 48.1168098, -31.5269642, 60.6732368, -85.6905441, 79.6437759
2: -13.3542852, 49.9176254, -16.7808247, 62.8047867, -76.1590729, 66.6984482
3: -28.4059334, 59.8230743, -35.6499596, 75.4741058, -103.8800354, 95.4730377
4: -17.0431519, 49.2884712, -21.4301624, 62.0332565, -79.0764084, 70.7186356

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1499414, upper bound: 71.1540665
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1481925, upper bound: 71.1540665
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -91.2265167, 172.5582886, -65.8488083, 123.1608505, -214.3873596, 238.4071045
1: -31.2533226, 60.1206589, -22.2078342, 42.6931381, -73.9464569, 82.3284912
2: -16.6439972, 62.2332649, -11.8839025, 44.2537460, -60.8977432, 74.1171570
3: -35.3476410, 74.7859116, -25.3091488, 53.0345116, -88.3821564, 100.0950623
4: -21.2512856, 61.4800644, -15.1839027, 43.6456490, -64.8969269, 76.6639557

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1484101, upper bound: 71.1499414
time: 0.70 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1484101, upper bound: 71.1549555
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -91.9336090, 174.0288239, -73.7855072, 139.1022339, -231.0358429, 247.8143311
1: -31.5269642, 60.6732368, -25.0173073, 48.1168098, -79.6437759, 85.6905441
2: -16.7808247, 62.8047867, -13.3542852, 49.9176254, -66.6984482, 76.1590729
3: -35.6499596, 75.4741058, -28.4059334, 59.8230743, -95.4730377, 103.8800354
4: -21.4301624, 62.0332565, -17.0431519, 49.2884712, -70.7186356, 79.0764084

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1540665, upper bound: 71.1499414
time: 0.78 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1540665, upper bound: 71.1549555
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -81.3959427, 152.7617340, -91.2265167, 172.5582886, -253.9542236, 243.9882050
1: -27.5621834, 52.6684456, -31.2533226, 60.1206589, -87.6828461, 83.9217529
2: -14.7725086, 54.6150055, -16.6439972, 62.2332649, -77.0057678, 71.2589798
3: -31.2484779, 65.5601120, -35.3476410, 74.7859116, -106.0343933, 100.9077530
4: -18.8214016, 54.0221977, -21.2512856, 61.4800644, -80.3014526, 75.2734604

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1500921, upper bound: 71.1501590
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1481925, upper bound: 71.1501590
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -88.6460800, 167.6033478, -91.9336090, 174.0288239, -262.6748962, 259.5369568
1: -30.2924213, 58.2035408, -31.5269642, 60.6732368, -90.9656601, 89.7305069
2: -16.1600780, 60.3205986, -16.7808247, 62.8047867, -78.9648590, 77.1014252
3: -34.2862015, 72.4371033, -35.6499596, 75.4741058, -109.7603073, 108.0870590
4: -20.6198254, 59.6320000, -21.4301624, 62.0332565, -82.6530838, 81.0621643

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1481925, upper bound: 71.1551732
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1500921, upper bound: 71.1551732
time: 0.80 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.72 seconds
NS_A1_B1_B1_A1, status: Status.VERIFIED, split count: 4, time: 3.72
Output dim: 4, lower bound: -71.1481925, upper bound: 71.1481925
NS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.72
Output dim: 4, lower bound: -71.1481925, upper bound: 71.1538509
NS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.72
Output dim: 4, lower bound: -71.1538509, upper bound: 71.1481925
NS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.72
Output dim: 4, lower bound: -71.1538509, upper bound: 71.1538509
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.72
Output dim: 4, lower bound: -71.1481925, upper bound: 71.1484101
NS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.72
Output dim: 4, lower bound: -71.1481925, upper bound: 71.1484101
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.72
Output dim: 4, lower bound: -71.1499414, upper bound: 71.1540665
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.72
Output dim: 4, lower bound: -71.1481925, upper bound: 71.1540665
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.72
Output dim: 4, lower bound: -71.1484101, upper bound: 71.1499414
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.72
Output dim: 4, lower bound: -71.1484101, upper bound: 71.1549555
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.72
Output dim: 4, lower bound: -71.1540665, upper bound: 71.1499414
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.72
Output dim: 4, lower bound: -71.1540665, upper bound: 71.1549555
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.72
Output dim: 4, lower bound: -71.1500921, upper bound: 71.1501590
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.72
Output dim: 4, lower bound: -71.1481925, upper bound: 71.1501590
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.72
Output dim: 4, lower bound: -71.1481925, upper bound: 71.1551732
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.72
Output dim: 4, lower bound: -71.1500921, upper bound: 71.1551732

## BFS NS instance: NS_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -73.7855072, 139.1022339, -65.8488083, 123.1608505, -196.9463501, 204.9510498
1: -25.0173073, 48.1168098, -22.2078342, 42.6931381, -67.7104416, 70.3246231
2: -13.3542852, 49.9176254, -11.8839025, 44.2537460, -57.6080246, 61.8015289
3: -28.4059334, 59.8230743, -25.3091488, 53.0345116, -81.4404449, 85.1322250
4: -17.0431519, 49.2884712, -15.1839027, 43.6456490, -60.6887932, 64.4723740

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_B1_A2_B1

### Relational analysis result of NS_A1_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1385995, upper bound: 71.1476137
time: 0.67 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2

### Relational analysis result of NS_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1481589, upper bound: 71.1539718
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -65.8488083, 123.1608505, -73.7855072, 139.1022339, -204.9510498, 196.9463501
1: -22.2078342, 42.6931381, -25.0173073, 48.1168098, -70.3246231, 67.7104416
2: -11.8839025, 44.2537460, -13.3542852, 49.9176254, -61.8015289, 57.6080246
3: -25.3091488, 53.0345116, -28.4059334, 59.8230743, -85.1322250, 81.4404449
4: -15.1839027, 43.6456490, -17.0431519, 49.2884712, -64.4723740, 60.6887932

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B2_A1_A1

### Relational analysis result of NS_A1_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1465585, upper bound: 71.1453605
time: 0.77 seconds

## Relational analysis of NS_A1_B1_B2_A1_A2

### Relational analysis result of NS_A1_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1465145, upper bound: 71.1465145
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -73.7855072, 139.1022339, -73.7855072, 139.1022339, -212.8877411, 212.8877411
1: -25.0173073, 48.1168098, -25.0173073, 48.1168098, -73.1341171, 73.1341095
2: -13.3542852, 49.9176254, -13.3542852, 49.9176254, -63.2719116, 63.2719040
3: -28.4059334, 59.8230743, -28.4059334, 59.8230743, -88.2290039, 88.2290039
4: -17.0431519, 49.2884712, -17.0431519, 49.2884712, -66.3316193, 66.3316193

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B2_A2_A1

### Relational analysis result of NS_A1_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1465585, upper bound: 71.1454667
time: 0.76 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2

### Relational analysis result of NS_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1465145, upper bound: 71.1524838
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -73.7855072, 139.1022339, -81.3959427, 152.7617340, -226.5472412, 220.4981689
1: -25.0173073, 48.1168098, -27.5621834, 52.6684456, -77.6857452, 75.6789856
2: -13.3542852, 49.9176254, -14.7725086, 54.6150055, -67.9692764, 64.6901245
3: -28.4059334, 59.8230743, -31.2484779, 65.5601120, -93.9660492, 91.0715485
4: -17.0431519, 49.2884712, -18.8214016, 54.0221977, -71.0653381, 68.1098709

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1453605, upper bound: 71.1525284
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2

### Relational analysis result of NS_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1472118, upper bound: 71.1524529
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -73.7855072, 139.1022339, -88.6460800, 167.6033478, -241.3888550, 227.7482758
1: -25.0173073, 48.1168098, -30.2924213, 58.2035408, -83.2208481, 78.4092255
2: -13.3542852, 49.9176254, -16.1600780, 60.3205986, -73.6748734, 66.0776978
3: -28.4059334, 59.8230743, -34.2862015, 72.4371033, -100.8430328, 94.1092758
4: -17.0431519, 49.2884712, -20.6198254, 59.6320000, -76.6751556, 69.9082947

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_B1

### Relational analysis result of NS_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1492032, upper bound: 71.1537541
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2

### Relational analysis result of NS_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1481589, upper bound: 71.1540665
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -81.3959427, 152.7617340, -65.8488083, 123.1608505, -204.5567932, 218.6105347
1: -27.5621834, 52.6684456, -22.2078342, 42.6931381, -70.2553253, 74.8762589
2: -14.7725086, 54.6150055, -11.8839025, 44.2537460, -59.0262489, 66.4988937
3: -31.2484779, 65.5601120, -25.3091488, 53.0345116, -84.2829895, 90.8692627
4: -18.8214016, 54.0221977, -15.1839027, 43.6456490, -62.4670486, 69.2060776

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_B1_A1_A1

### Relational analysis result of NS_A2_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1469621, upper bound: 71.1484589
time: 0.78 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2

### Relational analysis result of NS_A2_B1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1464933, upper bound: 71.1473371
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -88.6460800, 167.6033478, -65.8488083, 123.1608505, -211.8069305, 233.4521484
1: -30.2924213, 58.2035408, -22.2078342, 42.6931381, -72.9855499, 80.4113541
2: -16.1600780, 60.3205986, -11.8839025, 44.2537460, -60.4138260, 72.2044830
3: -34.2862015, 72.4371033, -25.3091488, 53.0345116, -87.3207092, 97.7462540
4: -20.6198254, 59.6320000, -15.1839027, 43.6456490, -64.2654724, 74.8158951

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_B1_A2_A1

### Relational analysis result of NS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1469621, upper bound: 71.1520226
time: 0.76 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2

### Relational analysis result of NS_A2_B1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1464933, upper bound: 71.1473371
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -81.3959427, 152.7617340, -73.7855072, 139.1022339, -220.4981689, 226.5472412
1: -27.5621834, 52.6684456, -25.0173073, 48.1168098, -75.6789856, 77.6857452
2: -14.7725086, 54.6150055, -13.3542852, 49.9176254, -64.6901245, 67.9692764
3: -31.2484779, 65.5601120, -28.4059334, 59.8230743, -91.0715485, 93.9660492
4: -18.8214016, 54.0221977, -17.0431519, 49.2884712, -68.1098709, 71.0653381

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_B2_A1_A1

### Relational analysis result of NS_A2_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1469621, upper bound: 71.1483848
time: 0.68 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2

### Relational analysis result of NS_A2_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1464933, upper bound: 71.1472118
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -88.6460800, 167.6033478, -73.7855072, 139.1022339, -227.7482605, 241.3888550
1: -30.2924213, 58.2035408, -25.0173073, 48.1168098, -78.4092255, 83.2208481
2: -16.1600780, 60.3205986, -13.3542852, 49.9176254, -66.0776978, 73.6748734
3: -34.2862015, 72.4371033, -28.4059334, 59.8230743, -94.1092758, 100.8430328
4: -20.6198254, 59.6320000, -17.0431519, 49.2884712, -69.9082947, 76.6751556

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_B2_A2_A1

### Relational analysis result of NS_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1480710, upper bound: 71.1504849
time: 0.82 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2

### Relational analysis result of NS_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1483765, upper bound: 71.1548341
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -81.3959427, 152.7617340, -81.3959427, 152.7617340, -234.1576843, 234.1576843
1: -27.5621834, 52.6684456, -27.5621834, 52.6684456, -80.2306213, 80.2306213
2: -14.7725086, 54.6150055, -14.7725086, 54.6150055, -69.3874969, 69.3874969
3: -31.2484779, 65.5601120, -31.2484779, 65.5601120, -96.8085938, 96.8085938
4: -18.8214016, 54.0221977, -18.8214016, 54.0221977, -72.8435669, 72.8435669

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1472755, upper bound: 71.1453417
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1464933, upper bound: 71.1471905
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -81.3959427, 152.7617340, -88.6460800, 167.6033478, -248.9992981, 241.4077759
1: -27.5621834, 52.6684456, -30.2924213, 58.2035408, -85.7657242, 82.9608536
2: -14.7725086, 54.6150055, -16.1600780, 60.3205986, -75.0930939, 70.7750626
3: -31.2484779, 65.5601120, -34.2862015, 72.4371033, -103.6855774, 99.8463135
4: -18.8214016, 54.0221977, -20.6198254, 59.6320000, -78.4533844, 74.6419983

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1482938, upper bound: 71.1483743
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1464933, upper bound: 71.1472024
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -88.6460800, 167.6033478, -81.3959427, 152.7617340, -241.4077606, 248.9992981
1: -30.2924213, 58.2035408, -27.5621834, 52.6684456, -82.9608536, 85.7657242
2: -16.1600780, 60.3205986, -14.7725086, 54.6150055, -70.7750626, 75.0931015
3: -34.2862015, 72.4371033, -31.2484779, 65.5601120, -99.8463135, 103.6855774
4: -20.6198254, 59.6320000, -18.8214016, 54.0221977, -74.6419983, 78.4533844

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1453512, upper bound: 71.1535547
time: 0.92 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2

### Relational analysis result of NS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1470800, upper bound: 71.1531404
time: 0.97 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -88.6460800, 167.6033478, -88.6460800, 167.6033478, -256.2493896, 256.2494202
1: -30.2924213, 58.2035408, -30.2924213, 58.2035408, -88.4959564, 88.4959564
2: -16.1600780, 60.3205986, -16.1600780, 60.3205986, -76.4806595, 76.4806595
3: -34.2862015, 72.4371033, -34.2862015, 72.4371033, -106.7233047, 106.7233047
4: -20.6198254, 59.6320000, -20.6198254, 59.6320000, -80.2518158, 80.2518158

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1465585, upper bound: 71.1522294
time: 0.86 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1470800, upper bound: 71.1531708
time: 0.74 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.23 seconds
NS_A1_B1_B1_A2_B1, status: Status.VERIFIED, split count: 5, time: 4.23
Output dim: 4, lower bound: -71.1385995, upper bound: 71.1476137
NS_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 4, lower bound: -71.1481589, upper bound: 71.1539718
NS_A1_B1_B2_A1_A1, status: Status.VERIFIED, split count: 5, time: 4.23
Output dim: 4, lower bound: -71.1465585, upper bound: 71.1453605
NS_A1_B1_B2_A1_A2, status: Status.VERIFIED, split count: 5, time: 4.23
Output dim: 4, lower bound: -71.1465145, upper bound: 71.1465145
NS_A1_B1_B2_A2_A1, status: Status.VERIFIED, split count: 5, time: 4.23
Output dim: 4, lower bound: -71.1465585, upper bound: 71.1454667
NS_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 4, lower bound: -71.1465145, upper bound: 71.1524838
NS_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 4, lower bound: -71.1453605, upper bound: 71.1525284
NS_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 4, lower bound: -71.1472118, upper bound: 71.1524529
NS_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 4, lower bound: -71.1492032, upper bound: 71.1537541
NS_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 4, lower bound: -71.1481589, upper bound: 71.1540665
NS_A2_B1_B1_A1_A1, status: Status.VERIFIED, split count: 5, time: 4.23
Output dim: 4, lower bound: -71.1469621, upper bound: 71.1484589
NS_A2_B1_B1_A1_A2, status: Status.VERIFIED, split count: 5, time: 4.23
Output dim: 4, lower bound: -71.1464933, upper bound: 71.1473371
NS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 4, lower bound: -71.1469621, upper bound: 71.1520226
NS_A2_B1_B1_A2_A2, status: Status.VERIFIED, split count: 5, time: 4.23
Output dim: 4, lower bound: -71.1464933, upper bound: 71.1473371
NS_A2_B1_B2_A1_A1, status: Status.VERIFIED, split count: 5, time: 4.23
Output dim: 4, lower bound: -71.1469621, upper bound: 71.1483848
NS_A2_B1_B2_A1_A2, status: Status.VERIFIED, split count: 5, time: 4.23
Output dim: 4, lower bound: -71.1464933, upper bound: 71.1472118
NS_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 4, lower bound: -71.1480710, upper bound: 71.1504849
NS_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 4, lower bound: -71.1483765, upper bound: 71.1548341
NS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.23
Output dim: 4, lower bound: -71.1472755, upper bound: 71.1453417
NS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.23
Output dim: 4, lower bound: -71.1464933, upper bound: 71.1471905
NS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.23
Output dim: 4, lower bound: -71.1482938, upper bound: 71.1483743
NS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.23
Output dim: 4, lower bound: -71.1464933, upper bound: 71.1472024
NS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 4, lower bound: -71.1453512, upper bound: 71.1535547
NS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 4, lower bound: -71.1470800, upper bound: 71.1531404
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 4, lower bound: -71.1465585, upper bound: 71.1522294
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.23
Output dim: 4, lower bound: -71.1470800, upper bound: 71.1531708

## BFS NS instance: NS_A1_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -73.7855072, 139.1022339, -61.2571335, 113.8663940, -187.6519012, 200.3593597
1: -25.0173073, 48.1168098, -20.5413780, 39.5949059, -64.6122055, 68.6581802
2: -13.3542852, 49.9176254, -11.0267153, 41.0238914, -54.3781700, 60.9443398
3: -28.4059334, 59.8230743, -23.5103874, 49.1691780, -77.5751114, 83.3334656
4: -17.0431519, 49.2884712, -14.0919380, 40.4716072, -57.5147514, 63.3804092

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B1_A2_B2_B1

### Relational analysis result of NS_A1_B1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1452377, upper bound: 71.1523927
time: 0.66 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_B2

### Relational analysis result of NS_A1_B1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1481519, upper bound: 71.1539561
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -68.4081573, 129.1255341, -73.7855072, 139.1022339, -207.5103912, 202.9110413
1: -23.2005672, 44.7357864, -25.0173073, 48.1168098, -71.3173599, 69.7530975
2: -12.3810244, 46.4367371, -13.3542852, 49.9176254, -62.2986488, 59.7910156
3: -26.4162025, 55.6006088, -28.4059334, 59.8230743, -86.2392731, 84.0065460
4: -15.8076572, 45.8427429, -17.0431519, 49.2884712, -65.0961304, 62.8858948

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1453776, upper bound: 71.1497420
time: 0.97 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1497472, upper bound: 71.1454667
time: 4.43 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -73.7855072, 139.1022339, -78.1272430, 147.0803070, -220.8658142, 217.2294769
1: -25.0173073, 48.1168098, -26.5070076, 50.8619461, -75.8792572, 74.6238174
2: -13.3542852, 49.9176254, -14.1301918, 52.5462112, -65.9004974, 64.0478210
3: -28.4059334, 59.8230743, -30.0567284, 63.1775818, -91.5835114, 89.8798065
4: -17.0431519, 49.2884712, -18.0706291, 51.8311234, -68.8742752, 67.3591003

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_B1_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1477124, upper bound: 71.1528471
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_B2

### Relational analysis result of NS_A1_B2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1453629, upper bound: 71.1529897
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -73.7855072, 139.1022339, -76.6951218, 144.0619812, -217.8474884, 215.7973175
1: -25.0173073, 48.1168098, -25.9950333, 49.7314301, -74.7487335, 74.1118469
2: -13.3542852, 49.9176254, -13.9296141, 51.5513306, -64.9056168, 63.8472176
3: -28.4059334, 59.8230743, -29.5056324, 61.8579178, -90.2638550, 89.3287048
4: -17.0431519, 49.2884712, -17.7518425, 50.9514656, -67.9946136, 67.0403137

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1441325, upper bound: 71.1484420
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1472118, upper bound: 71.1524529
time: 0.97 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -69.0932922, 130.0623779, -81.7319870, 156.1461639, -225.2394562, 211.7943726
1: -23.3626347, 45.0348969, -27.9740181, 54.0183678, -77.3810043, 73.0089111
2: -12.4840479, 46.6827164, -14.9771605, 55.7253456, -68.2093964, 61.6598740
3: -26.5788536, 55.9599380, -31.9204483, 67.3245316, -93.9033813, 87.8803635
4: -15.9552498, 46.0399818, -19.2586079, 55.0077095, -70.9629440, 65.2985840

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_B1_B1

### Relational analysis result of NS_A1_B2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1493020, upper bound: 71.1528693
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_B2

### Relational analysis result of NS_A1_B2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1501771, upper bound: 71.1533452
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -73.7855072, 139.1022339, -84.1622314, 158.7486420, -232.5341492, 223.2644501
1: -25.0173073, 48.1168098, -28.6183739, 54.9416695, -79.9589691, 76.7351837
2: -13.3542852, 49.9176254, -15.3075724, 57.0017242, -70.3560104, 65.2251968
3: -28.4059334, 59.8230743, -32.4546890, 68.3871613, -96.7930908, 92.2777634
4: -17.0431519, 49.2884712, -19.5165005, 56.4163742, -73.4595261, 68.8049698

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_B2_B1

### Relational analysis result of NS_A1_B2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1453605, upper bound: 71.1528902
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2_B2

### Relational analysis result of NS_A1_B2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1499376, upper bound: 71.1524771
time: 1.41 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -86.4461136, 164.4075775, -65.8488083, 123.1608505, -209.6069489, 230.2563782
1: -29.5339794, 56.7140808, -22.2078342, 42.6931381, -72.2271194, 78.9218979
2: -15.7065840, 58.7480125, -11.8839025, 44.2537460, -59.9603310, 70.6319046
3: -33.4223480, 70.4806900, -25.3091488, 53.0345116, -86.4568634, 95.7898407
4: -20.0807629, 58.0298615, -15.1839027, 43.6456490, -63.7264099, 73.2137604

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_B1_A2_A1_B1

### Relational analysis result of NS_A2_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1462008, upper bound: 71.1518773
time: 0.86 seconds

## Relational analysis of NS_A2_B1_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B1_A2_A1_A1

### Relational analysis result of NS_A2_B1_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1459459, upper bound: 71.1486750
time: 0.71 seconds

## Relational analysis of NS_A2_B1_B1_A2_A1_A2

### Relational analysis result of NS_A2_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1468390, upper bound: 71.1519888
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -81.7319870, 156.1461639, -69.0932922, 130.0623779, -211.7943726, 225.2394562
1: -27.9740181, 54.0183678, -23.3626347, 45.0348969, -73.0089111, 77.3810043
2: -14.9771605, 55.7253456, -12.4840479, 46.6827164, -61.6598740, 68.2093964
3: -31.9204483, 67.3245316, -26.5788536, 55.9599380, -87.8803635, 93.9033813
4: -19.2586079, 55.0077095, -15.9552498, 46.0399818, -65.2985840, 70.9629440

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B2_A2_A1_A1

### Relational analysis result of NS_A2_B1_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1473756, upper bound: 71.1493104
time: 0.86 seconds

## Relational analysis of NS_A2_B1_B2_A2_A1_A2

### Relational analysis result of NS_A2_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1488471, upper bound: 71.1504623
time: 0.99 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -84.1622314, 158.7486420, -73.7855072, 139.1022339, -223.2644501, 232.5341492
1: -28.6183739, 54.9416695, -25.0173073, 48.1168098, -76.7351837, 79.9589691
2: -15.3075724, 57.0017242, -13.3542852, 49.9176254, -65.2251968, 70.3560104
3: -32.4546890, 68.3871613, -28.4059334, 59.8230743, -92.2777634, 96.7930908
4: -19.5165005, 56.4163742, -17.0431519, 49.2884712, -68.8049698, 73.4595261

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_B2_A2_A2_A1

### Relational analysis result of NS_A2_B1_B2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1470709, upper bound: 71.1454667
time: 0.83 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2_A2

### Relational analysis result of NS_A2_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1499527, upper bound: 71.1531386
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -88.6460800, 167.6033478, -78.1272430, 147.0803070, -235.7263641, 245.7305908
1: -30.2924213, 58.2035408, -26.5070076, 50.8619461, -81.1543655, 84.7105484
2: -16.1600780, 60.3205986, -14.1301918, 52.5462112, -68.7062912, 74.4507904
3: -34.2862015, 72.4371033, -30.0567284, 63.1775818, -97.4637833, 102.4938278
4: -20.6198254, 59.6320000, -18.0706291, 51.8311234, -72.4509506, 77.7026215

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B1_B1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1458147, upper bound: 71.1519878
time: 0.84 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_B2

### Relational analysis result of NS_A2_B2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1471531, upper bound: 71.1534068
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -88.6460800, 167.6033478, -76.6951218, 144.0619812, -232.7080231, 244.2984467
1: -30.2924213, 58.2035408, -25.9950333, 49.7314301, -80.0238495, 84.1985779
2: -16.1600780, 60.3205986, -13.9296141, 51.5513306, -67.7114105, 74.2502060
3: -34.2862015, 72.4371033, -29.5056324, 61.8579178, -96.1441193, 101.9427338
4: -20.6198254, 59.6320000, -17.7518425, 50.9514656, -71.5712814, 77.3838272

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1465052, upper bound: 71.1521561
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1465052, upper bound: 71.1531404
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -86.4461136, 164.4075775, -88.6460800, 167.6033478, -254.0494232, 253.0536346
1: -29.5339794, 56.7140808, -30.2924213, 58.2035408, -87.7375183, 87.0065002
2: -15.7065840, 58.7480125, -16.1600780, 60.3205986, -76.0271606, 74.9080811
3: -33.4223480, 70.4806900, -34.2862015, 72.4371033, -105.8594513, 104.7668915
4: -20.0807629, 58.0298615, -20.6198254, 59.6320000, -79.7127609, 78.6496887

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1425412, upper bound: 71.1453664
time: 0.92 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1490207, upper bound: 71.1522122
time: 0.89 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1488237, upper bound: 71.1496629
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -82.7067871, 156.2546234, -88.6460800, 167.6033478, -250.3101196, 244.9006805
1: -28.1770096, 54.0810471, -30.2924213, 58.2035408, -86.3805542, 84.3734665
2: -15.0552874, 56.1127815, -16.1600780, 60.3205986, -75.3758850, 72.2728500
3: -31.9458065, 67.2924957, -34.2862015, 72.4371033, -104.3829117, 101.5786972
4: -19.1932411, 55.5088654, -20.6198254, 59.6320000, -78.8252411, 76.1286926

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1450552, upper bound: 71.1472560
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1451147, upper bound: 71.1475800
time: 1.01 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 5.08 seconds
NS_A1_B1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 4, lower bound: -71.1452377, upper bound: 71.1523927
NS_A1_B1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 4, lower bound: -71.1481519, upper bound: 71.1539561
NS_A1_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 4, lower bound: -71.1453776, upper bound: 71.1497420
NS_A1_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 4, lower bound: -71.1497472, upper bound: 71.1454667
NS_A1_B2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 4, lower bound: -71.1477124, upper bound: 71.1528471
NS_A1_B2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 4, lower bound: -71.1453629, upper bound: 71.1529897
NS_A1_B2_A2_B1_B2_A1, status: Status.VERIFIED, split count: 6, time: 5.08
Output dim: 4, lower bound: -71.1441325, upper bound: 71.1484420
NS_A1_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 4, lower bound: -71.1472118, upper bound: 71.1524529
NS_A1_B2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 4, lower bound: -71.1493020, upper bound: 71.1528693
NS_A1_B2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 4, lower bound: -71.1501771, upper bound: 71.1533452
NS_A1_B2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 4, lower bound: -71.1453605, upper bound: 71.1528902
NS_A1_B2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 4, lower bound: -71.1499376, upper bound: 71.1524771
NS_A2_B1_B1_A2_A1_A1, status: Status.VERIFIED, split count: 6, time: 5.08
Output dim: 4, lower bound: -71.1459459, upper bound: 71.1486750
NS_A2_B1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 4, lower bound: -71.1468390, upper bound: 71.1519888
NS_A2_B1_B2_A2_A1_A1, status: Status.VERIFIED, split count: 6, time: 5.08
Output dim: 4, lower bound: -71.1473756, upper bound: 71.1493104
NS_A2_B1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 4, lower bound: -71.1488471, upper bound: 71.1504623
NS_A2_B1_B2_A2_A2_A1, status: Status.VERIFIED, split count: 6, time: 5.08
Output dim: 4, lower bound: -71.1470709, upper bound: 71.1454667
NS_A2_B1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 4, lower bound: -71.1499527, upper bound: 71.1531386
NS_A2_B2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 4, lower bound: -71.1458147, upper bound: 71.1519878
NS_A2_B2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 4, lower bound: -71.1471531, upper bound: 71.1534068
NS_A2_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 4, lower bound: -71.1465052, upper bound: 71.1521561
NS_A2_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 4, lower bound: -71.1465052, upper bound: 71.1531404
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 4, lower bound: -71.1490207, upper bound: 71.1522122
NS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.08
Output dim: 4, lower bound: -71.1488237, upper bound: 71.1496629
NS_A2_B2_A2_B2_A2_A1, status: Status.VERIFIED, split count: 6, time: 5.08
Output dim: 4, lower bound: -71.1450552, upper bound: 71.1472560
NS_A2_B2_A2_B2_A2_A2, status: Status.VERIFIED, split count: 6, time: 5.08
Output dim: 4, lower bound: -71.1451147, upper bound: 71.1475800

## BFS NS instance: NS_A1_B1_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -70.8617554, 133.4181824, -47.6339836, 88.4396744, -159.3014221, 181.0521698
1: -23.9928131, 46.2243118, -16.0081577, 31.1225796, -55.1153870, 62.2324677
2: -12.8093109, 47.9465828, -8.5073566, 32.2112350, -45.0205460, 56.4539375
3: -27.2543049, 57.4349976, -18.2406483, 38.4641418, -65.7184448, 75.6756439
4: -16.3586750, 47.3155403, -10.9254990, 31.5893764, -47.9480515, 58.2410393

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B1_A2_B2_B1_A1

### Relational analysis result of NS_A1_B1_B1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1433639, upper bound: 71.1455562
time: 0.86 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_B1_A2

### Relational analysis result of NS_A1_B1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1433320, upper bound: 71.1506299
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -73.7855072, 139.1022339, -57.2370224, 106.1911240, -179.9766235, 196.3392639
1: -25.0173073, 48.1168098, -19.1456184, 36.9419403, -61.9592476, 67.2624283
2: -13.3542852, 49.9176254, -10.2821445, 38.2947044, -51.6489868, 60.1997643
3: -28.4059334, 59.8230743, -21.9694061, 45.8508377, -74.2567749, 81.7924805
4: -17.0431519, 49.2884712, -13.1389627, 37.7668228, -54.8099670, 62.4274330

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B1_A2_B2_B2_B1

### Relational analysis result of NS_A1_B1_B1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1438174, upper bound: 71.1526347
time: 0.82 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_B2_B2

### Relational analysis result of NS_A1_B1_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1463183, upper bound: 71.1524789
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -68.3772964, 129.0691528, -68.2443542, 129.3410645, -197.7183228, 197.3134918
1: -23.1901970, 44.7162552, -23.1020164, 44.6918716, -67.8820648, 67.8182678
2: -12.3754358, 46.4169121, -12.3300610, 46.5189171, -58.8943520, 58.7469711
3: -26.4048004, 55.5763474, -26.3570194, 55.5973167, -82.0021210, 81.9333572
4: -15.8006153, 45.8230858, -15.7524538, 45.9710922, -61.7717056, 61.5755386

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_B2_A2_A2_B1_B1

### Relational analysis result of NS_A1_B1_B2_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1445788, upper bound: 71.1489069
time: 0.89 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2_B1_B2

### Relational analysis result of NS_A1_B1_B2_A2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1443289, upper bound: 71.1474957
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -68.4081573, 129.1255341, -71.4678955, 134.8689880, -203.2771454, 200.5934296
1: -23.2005672, 44.7357864, -24.2386436, 46.6459732, -69.8465271, 68.9744263
2: -12.3810244, 46.4367371, -12.9322739, 48.4183350, -60.7993584, 59.3690109
3: -26.4162025, 55.6006088, -27.5348587, 57.9817047, -84.3979034, 83.1354675
4: -15.8076572, 45.8427429, -16.5068722, 47.7892151, -63.5968704, 62.3496132

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B1

### Relational analysis result of NS_A1_B1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1454089, upper bound: 71.1524838
time: 0.74 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B2

### Relational analysis result of NS_A1_B1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1454089, upper bound: 71.1524838
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -69.0932922, 130.0623779, -71.6832428, 137.2845306, -206.3778229, 201.7456207
1: -23.3626347, 45.0348969, -24.4180317, 46.7670250, -70.1296616, 69.4529190
2: -12.4840479, 46.6827164, -13.1171646, 48.2465630, -60.7306099, 59.7998810
3: -26.5788536, 55.9599380, -27.8897953, 58.3054657, -84.8843231, 83.8497162
4: -15.9552498, 46.0399818, -16.8491478, 47.6110001, -63.5662384, 62.8891296

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1377481, upper bound: 71.1523062
time: 0.94 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_B2

### Relational analysis result of NS_A1_B2_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1476766, upper bound: 71.1524152
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -73.7855072, 139.1022339, -74.5900726, 140.2841187, -214.0696259, 213.6922760
1: -25.0173073, 48.1168098, -25.2587299, 48.5457878, -73.5630875, 73.3755341
2: -13.3542852, 49.9176254, -13.4757414, 50.1595955, -63.5138817, 63.3933640
3: -28.4059334, 59.8230743, -28.6995621, 60.2677155, -88.6736450, 88.5226364
4: -17.0431519, 49.2884712, -17.2441788, 49.4638138, -66.5069656, 66.5326538

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_B1_B2_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1438174, upper bound: 71.1510727
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_B2_B2

### Relational analysis result of NS_A1_B2_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1453270, upper bound: 71.1528437
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -71.4678955, 134.8689880, -76.6951218, 144.0619812, -215.5298767, 211.5641022
1: -24.2386436, 46.6459732, -25.9950333, 49.7314301, -73.9700623, 72.6409988
2: -12.9322739, 48.4183350, -13.9296141, 51.5513306, -64.4836044, 62.3479424
3: -27.5348587, 57.9817047, -29.5056324, 61.8579178, -89.3927765, 87.4873276
4: -16.5068722, 47.7892151, -17.7518425, 50.9514656, -67.4583282, 65.5410461

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1472118, upper bound: 71.1454454
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1453629, upper bound: 71.1524529
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -66.3565826, 124.6176071, -67.1611099, 127.1303253, -193.4869080, 191.7787170
1: -22.3783703, 43.2116547, -22.6353550, 43.5189095, -65.8972778, 65.8470001
2: -11.9649763, 44.7848701, -12.1209774, 45.2287750, -57.1937485, 56.9058456
3: -25.4797096, 53.6649933, -25.7805500, 54.2148132, -79.6945190, 79.4455261
4: -15.3039875, 44.1526604, -15.5110331, 44.7341423, -60.0381317, 59.6636887

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_B1_B1_B1

### Relational analysis result of NS_A1_B2_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1468508, upper bound: 71.1503842
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_B1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1485060, upper bound: 71.1514617
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1481020, upper bound: 71.1466427
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -69.0932922, 130.0623779, -77.7428894, 148.8098450, -217.9031372, 207.8052673
1: -23.3626347, 45.0348969, -26.5909290, 51.2852364, -74.6478729, 71.6258240
2: -12.4840479, 46.6827164, -14.2547655, 52.9671364, -65.4511871, 60.9374809
3: -26.5788536, 55.9599380, -30.3973598, 63.9421768, -90.5210266, 86.3572922
4: -15.9552498, 46.0399818, -18.3203106, 52.3190193, -68.2742615, 64.3602905

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_B1_B2_B1

### Relational analysis result of NS_A1_B2_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1484885, upper bound: 71.1522447
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_B1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1492296, upper bound: 71.1519325
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1484747, upper bound: 71.1470802
time: 1.51 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -73.7855072, 139.1022339, -82.3133698, 156.5305328, -230.3160400, 221.4156036
1: -25.0173073, 48.1168098, -28.0891361, 54.0074615, -79.0247650, 76.2059402
2: -13.3542852, 49.9176254, -14.9534559, 55.9817657, -69.3360519, 64.8710785
3: -28.4059334, 59.8230743, -31.8437176, 67.1011276, -95.5070572, 91.6667938
4: -17.0431519, 49.2884712, -19.1258888, 55.2833595, -72.3265076, 68.4143600

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_B2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1478063, upper bound: 71.1454574
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1478063, upper bound: 71.1524771
time: 0.94 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -73.7855072, 139.1022339, -78.3574066, 147.7373962, -221.5229034, 217.4596252
1: -25.0173073, 48.1168098, -26.5852623, 51.0275879, -76.0448761, 74.7020721
2: -13.3542852, 49.9176254, -14.2363129, 53.0023880, -66.3566742, 64.1539383
3: -28.4059334, 59.8230743, -30.2151489, 63.5047493, -91.9106674, 90.0382233
4: -17.0431519, 49.2884712, -18.1454506, 52.4693222, -69.5124741, 67.4339218

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_B2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1490599, upper bound: 71.1493860
time: 0.96 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1496046, upper bound: 71.1524771
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A1_A2

### Backsubstitution after applying NS history:
0: -81.4499207, 155.3026428, -65.8488083, 123.1608505, -204.6107788, 221.1514587
1: -27.8734722, 53.6046371, -22.2078342, 42.6931381, -70.5666046, 75.8124466
2: -14.8045921, 55.5911636, -11.8839025, 44.2537460, -59.0583382, 67.4750671
3: -31.5679092, 66.5957794, -25.3091488, 53.0345116, -84.6024170, 91.9049301
4: -18.9398422, 54.8525467, -15.1839027, 43.6456490, -62.5854836, 70.0364304

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_B1_A2_A1_A2_A1

### Relational analysis result of NS_A2_B1_B1_A2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1461884, upper bound: 71.1489041
time: 0.82 seconds

## Relational analysis of NS_A2_B1_B1_A2_A1_A2_A2

### Relational analysis result of NS_A2_B1_B1_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1466788, upper bound: 71.1507905
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A1_A2

### Backsubstitution after applying NS history:
0: -77.7428894, 148.8098450, -69.0932922, 130.0623779, -207.8052673, 217.9031372
1: -26.5909290, 51.2852364, -23.3626347, 45.0348969, -71.6258240, 74.6478729
2: -14.2547655, 52.9671364, -12.4840479, 46.6827164, -60.9374809, 65.4511871
3: -30.3973598, 63.9421768, -26.5788536, 55.9599380, -86.3572922, 90.5210266
4: -18.3203106, 52.3190193, -15.9552498, 46.0399818, -64.3602905, 68.2742615

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_B2_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_B2_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_B2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_B2_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B2_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_B2_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_B2_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_B2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_B2_A2_A1_A2_B1

### Relational analysis result of NS_A2_B1_B2_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1474226, upper bound: 71.1495266
time: 0.88 seconds

## Relational analysis of NS_A2_B1_B2_A2_A1_A2_B2

### Relational analysis result of NS_A2_B1_B2_A2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1464998, upper bound: 71.1494613
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -78.3574066, 147.7373962, -73.7855072, 139.1022339, -217.4596252, 221.5229034
1: -26.5852623, 51.0275879, -25.0173073, 48.1168098, -74.7020721, 76.0448761
2: -14.2363129, 53.0023880, -13.3542852, 49.9176254, -64.1539383, 66.3566742
3: -30.2151489, 63.5047493, -28.4059334, 59.8230743, -90.0382233, 91.9106674
4: -18.1454506, 52.4693222, -17.0431519, 49.2884712, -67.4339218, 69.5124741

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_B2_A2_A2_A2_B1

### Relational analysis result of NS_A2_B1_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1453512, upper bound: 71.1531386
time: 0.86 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2_A2_B2

### Relational analysis result of NS_A2_B1_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1454359, upper bound: 71.1531386
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -83.2681198, 156.8382263, -71.6832428, 137.2845306, -220.5526428, 228.5214691
1: -28.2571011, 54.2695580, -24.4180317, 46.7670250, -75.0241013, 78.6875916
2: -15.1173201, 56.2637787, -13.1171646, 48.2465630, -63.3638840, 69.3809357
3: -32.0387726, 67.5293961, -27.8897953, 58.3054657, -90.3442230, 95.4191742
4: -19.2889595, 55.6473083, -16.8491478, 47.6110001, -66.8999557, 72.4964600

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1_B1_B1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1454696, upper bound: 71.1517470
time: 0.96 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_B1_B2

### Relational analysis result of NS_A2_B2_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1454366, upper bound: 71.1515869
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -88.6460800, 167.6033478, -74.5900726, 140.2841187, -228.9301910, 242.1934052
1: -30.2924213, 58.2035408, -25.2587299, 48.5457878, -78.8381882, 83.4622650
2: -16.1600780, 60.3205986, -13.4757414, 50.1595955, -66.3196716, 73.7963333
3: -34.2862015, 72.4371033, -28.6995621, 60.2677155, -94.5539169, 101.1366577
4: -20.6198254, 59.6320000, -17.2441788, 49.4638138, -70.0836411, 76.8761749

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B1_B1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1471531, upper bound: 71.1516529
time: 0.95 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1471531, upper bound: 71.1534068
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -86.4461136, 164.4075775, -76.6951218, 144.0619812, -230.5080719, 241.1026764
1: -29.5339794, 56.7140808, -25.9950333, 49.7314301, -79.2654114, 82.7091141
2: -15.7065840, 58.7480125, -13.9296141, 51.5513306, -67.2579117, 72.6776276
3: -33.4223480, 70.4806900, -29.5056324, 61.8579178, -95.2802582, 99.9863205
4: -20.0807629, 58.0298615, -17.7518425, 50.9514656, -71.0322266, 75.7817001

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1423663, upper bound: 71.1521373
time: 0.93 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_A1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1432606, upper bound: 71.1503884
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_A2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1453918, upper bound: 71.1506749
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -82.7067871, 156.2546234, -76.6951218, 144.0619812, -226.7687531, 232.9497223
1: -28.1770096, 54.0810471, -25.9950333, 49.7314301, -77.9084396, 80.0760803
2: -15.0552874, 56.1127815, -13.9296141, 51.5513306, -66.6066208, 70.0423965
3: -31.9458065, 67.2924957, -29.5056324, 61.8579178, -93.8037262, 96.7981262
4: -19.1932411, 55.5088654, -17.7518425, 50.9514656, -70.1446991, 73.2607117

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1442901, upper bound: 71.1527515
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_B2_A2_A1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1455732, upper bound: 71.1515046
time: 0.98 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A2_A2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1431115, upper bound: 71.1510096
time: 1.45 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -86.4461136, 164.4075775, -86.2317505, 162.8605652, -249.3066559, 250.6393127
1: -29.5339794, 56.7140808, -29.4165878, 56.4552879, -85.9892654, 86.1306686
2: -15.7065840, 58.7480125, -15.7019749, 58.5120316, -74.2186127, 74.4499893
3: -33.4223480, 70.4806900, -33.3013191, 70.2293320, -103.6516800, 103.7820053
4: -20.0807629, 58.0298615, -20.0258121, 57.8531189, -77.9338837, 78.0556717

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1482164, upper bound: 71.1490908
time: 0.86 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1486113, upper bound: 71.1506841
time: 0.96 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 4.57 seconds
NS_A1_B1_B1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1433639, upper bound: 71.1455562
NS_A1_B1_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1433320, upper bound: 71.1506299
NS_A1_B1_B1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1438174, upper bound: 71.1526347
NS_A1_B1_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1463183, upper bound: 71.1524789
NS_A1_B1_B2_A2_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1445788, upper bound: 71.1489069
NS_A1_B1_B2_A2_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1443289, upper bound: 71.1474957
NS_A1_B1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1454089, upper bound: 71.1524838
NS_A1_B1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1454089, upper bound: 71.1524838
NS_A1_B2_A2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1377481, upper bound: 71.1523062
NS_A1_B2_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1476766, upper bound: 71.1524152
NS_A1_B2_A2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1438174, upper bound: 71.1510727
NS_A1_B2_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1453270, upper bound: 71.1528437
NS_A1_B2_A2_B1_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1472118, upper bound: 71.1454454
NS_A1_B2_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1453629, upper bound: 71.1524529
NS_A1_B2_A2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1485060, upper bound: 71.1514617
NS_A1_B2_A2_B2_B1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1481020, upper bound: 71.1466427
NS_A1_B2_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1492296, upper bound: 71.1519325
NS_A1_B2_A2_B2_B1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1484747, upper bound: 71.1470802
NS_A1_B2_A2_B2_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1478063, upper bound: 71.1454574
NS_A1_B2_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1478063, upper bound: 71.1524771
NS_A1_B2_A2_B2_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1490599, upper bound: 71.1493860
NS_A1_B2_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1496046, upper bound: 71.1524771
NS_A2_B1_B1_A2_A1_A2_A1, status: Status.VERIFIED, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1461884, upper bound: 71.1489041
NS_A2_B1_B1_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1466788, upper bound: 71.1507905
NS_A2_B1_B2_A2_A1_A2_B1, status: Status.VERIFIED, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1474226, upper bound: 71.1495266
NS_A2_B1_B2_A2_A1_A2_B2, status: Status.VERIFIED, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1464998, upper bound: 71.1494613
NS_A2_B1_B2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1453512, upper bound: 71.1531386
NS_A2_B1_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1454359, upper bound: 71.1531386
NS_A2_B2_A2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1454696, upper bound: 71.1517470
NS_A2_B2_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1454366, upper bound: 71.1515869
NS_A2_B2_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1471531, upper bound: 71.1516529
NS_A2_B2_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1471531, upper bound: 71.1534068
NS_A2_B2_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1432606, upper bound: 71.1503884
NS_A2_B2_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1453918, upper bound: 71.1506749
NS_A2_B2_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1455732, upper bound: 71.1515046
NS_A2_B2_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1431115, upper bound: 71.1510096
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1482164, upper bound: 71.1490908
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.57
Output dim: 4, lower bound: -71.1486113, upper bound: 71.1506841

## BFS NS instance: NS_A1_B1_B1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -65.3013687, 123.1984100, -47.6339836, 88.4396744, -153.7410431, 170.8323975
1: -22.1351128, 42.7695847, -16.0081577, 31.1225796, -53.2576904, 58.7777405
2: -11.8052397, 44.3752518, -8.5073566, 32.2112350, -44.0164757, 52.8826065
3: -25.1982040, 53.1005516, -18.2406483, 38.4641418, -63.6623459, 71.3411789
4: -15.0787373, 43.7572556, -10.9254990, 31.5893764, -46.6681099, 54.6827545

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B1_A2_B2_B1_A2_B1

### Relational analysis result of NS_A1_B1_B1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1433320, upper bound: 71.1506299
time: 0.76 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_B1_A2_B2

### Relational analysis result of NS_A1_B1_B1_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1433320, upper bound: 71.1506299
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -73.7855072, 139.1022339, -53.1349792, 99.2397079, -173.0252075, 192.2371979
1: -25.0173073, 48.1168098, -17.8659153, 34.5999603, -59.6172638, 65.9827118
2: -13.3542852, 49.9176254, -9.5169573, 35.7835045, -49.1377831, 59.4345818
3: -28.4059334, 59.8230743, -20.4746952, 42.7898178, -71.1957550, 80.2977676
4: -17.0431519, 49.2884712, -12.2092848, 35.1428452, -52.1859894, 61.4977570

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_B1_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_B1_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_B1_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B1_A2_B2_B2_B1_A1

### Relational analysis result of NS_A1_B1_B1_A2_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1433320, upper bound: 71.1455920
time: 0.91 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_B2_B1_A2

### Relational analysis result of NS_A1_B1_B1_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1453270, upper bound: 71.1524789
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -73.7855072, 139.1022339, -53.1138725, 98.3912430, -172.1767578, 192.2161102
1: -25.0173073, 48.1168098, -17.7447529, 34.3278999, -59.3452034, 65.8615494
2: -13.3542852, 49.9176254, -9.5283327, 35.5611610, -48.9154434, 59.4459534
3: -28.4059334, 59.8230743, -20.4099407, 42.5577011, -70.9636383, 80.2330170
4: -17.0431519, 49.2884712, -12.1856918, 35.0390244, -52.0821648, 61.4741631

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_B1_A2_B2_B2_B2_B1

### Relational analysis result of NS_A1_B1_B1_A2_B2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1423314, upper bound: 71.1480196
time: 0.86 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_B2_B2_B2

### Relational analysis result of NS_A1_B1_B1_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1463183, upper bound: 71.1524789
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -68.4081573, 129.1255341, -66.5289612, 126.3369217, -194.7450562, 195.6544800
1: -23.2005672, 44.7357864, -22.6714611, 43.7581329, -66.9586868, 67.4072495
2: -12.3810244, 46.4367371, -12.0287647, 45.3874512, -57.7684746, 58.4654961
3: -26.4162025, 55.6006088, -25.7600098, 54.2753029, -80.6914902, 81.3606110
4: -15.8076572, 45.8427429, -15.4071035, 44.6459045, -60.4535599, 61.2498474

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B1_B1

### Relational analysis result of NS_A1_B1_B2_A2_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1434389, upper bound: 71.1462878
time: 0.79 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 12

## BFS NS instance: NS_A1_B1_B2_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -68.4081573, 129.1255341, -66.0960770, 124.9009933, -193.3091431, 195.2215881
1: -23.2005672, 44.7357864, -22.4228764, 43.2689667, -66.4695206, 67.1586609
2: -12.3810244, 46.4367371, -11.9592190, 44.9382896, -57.3193130, 58.3959541
3: -26.4162025, 55.6006088, -25.5470219, 53.7612648, -80.1774673, 81.1476212
4: -15.8076572, 45.8427429, -15.2724581, 44.3456650, -60.1533203, 61.1152000

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B2_A1

### Relational analysis result of NS_A1_B1_B2_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1412360, upper bound: 71.1515160
time: 1.67 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2_B2_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1453538, upper bound: 71.1524838
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_B1_B1

### Backsubstitution after applying NS history:
0: -66.3565826, 124.6176071, -55.1105690, 105.4298553, -171.7864380, 179.7281799
1: -22.3783703, 43.2116547, -18.7954273, 36.1542282, -58.5325928, 62.0070686
2: -11.9649763, 44.7848701, -10.0173635, 37.5091858, -49.4741592, 54.8022232
3: -25.4797096, 53.6649933, -21.3924236, 44.9649696, -70.4446793, 75.0574188
4: -15.3039875, 44.1526604, -12.8451662, 36.9872589, -52.2912445, 56.9978256

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_B1_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1452313, upper bound: 71.1500527
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_B1_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1440715, upper bound: 71.1512520
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_B1_B2

### Relational analysis result of NS_A1_B2_A2_B1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1471692, upper bound: 71.1519777
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_B1_B2

### Backsubstitution after applying NS history:
0: -69.0932922, 130.0623779, -68.7347946, 132.0270386, -201.1203308, 198.7971802
1: -23.3626347, 45.0348969, -23.4756031, 44.9308968, -68.2935333, 68.5104904
2: -12.4840479, 46.6827164, -12.6002512, 46.3624878, -58.8465309, 59.2829666
3: -26.5788536, 55.9599380, -26.8194199, 56.0100746, -82.5889282, 82.7793350
4: -15.9552498, 46.0399818, -16.1925220, 45.7353020, -61.6905518, 62.2325020

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_B2_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1453178, upper bound: 71.1500663
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 49

## BFS NS instance: NS_A1_B2_A2_B1_B1_B2_B1

### Backsubstitution after applying NS history:
0: -70.8617554, 133.4181824, -58.9640961, 110.3200989, -181.1818390, 192.3822784
1: -23.9928131, 46.2243118, -19.9102821, 38.5166664, -62.5094757, 66.1345978
2: -12.8093109, 47.9465828, -10.5833797, 39.8637505, -52.6730614, 58.5299606
3: -27.2543049, 57.4349976, -22.6200924, 47.7307014, -74.9850082, 80.0550919
4: -16.3586750, 47.3155403, -13.5778618, 39.2038002, -55.5624771, 60.8934021

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1_B1_B2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1473776, upper bound: 71.1463958
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_B2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1473776, upper bound: 71.1523269
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_B2_B2

### Backsubstitution after applying NS history:
0: -73.7855072, 139.1022339, -71.2987595, 134.1162109, -207.9017181, 210.4009857
1: -25.0173073, 48.1168098, -24.1379108, 46.4130898, -71.4303970, 72.2547073
2: -13.3542852, 49.9176254, -12.8685703, 47.9549179, -61.3091965, 62.7861938
3: -28.4059334, 59.8230743, -27.4456234, 57.5960274, -86.0019608, 87.2686996
4: -17.0431519, 49.2884712, -16.4754963, 47.2507591, -64.2939148, 65.7639694

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_B1_B2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1441663, upper bound: 71.1486626
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_B2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1477938, upper bound: 71.1528437
time: 0.99 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -66.0960770, 124.9009933, -76.6951218, 144.0619812, -210.1580505, 201.5960693
1: -22.4228764, 43.2689667, -25.9950333, 49.7314301, -72.1542969, 69.2639999
2: -11.9592190, 44.9382896, -13.9296141, 51.5513306, -63.5105515, 58.8678970
3: -25.5470219, 53.7612648, -29.5056324, 61.8579178, -87.4049225, 83.2668991
4: -15.2724581, 44.3456650, -17.7518425, 50.9514656, -66.2239151, 62.0975075

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A2_A1

### Relational analysis result of NS_A1_B2_A2_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1457050, upper bound: 71.1510024
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 23

## BFS NS instance: NS_A1_B2_A2_B2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -64.2775574, 120.7818680, -67.1611099, 127.1303253, -191.4078827, 187.9429779
1: -21.6878128, 41.9143753, -22.6353550, 43.5189095, -65.2067261, 64.5497131
2: -11.5903749, 43.4314423, -12.1209774, 45.2287750, -56.8191452, 55.5524216
3: -24.7029476, 52.0356293, -25.7805500, 54.2148132, -78.9177628, 77.8161774
4: -14.8279448, 42.7866287, -15.5110331, 44.7341423, -59.5620880, 58.2976570

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 49

## BFS NS instance: NS_A1_B2_A2_B2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -67.1225586, 126.3770218, -77.7428894, 148.8098450, -215.9324036, 204.1199036
1: -22.6893063, 43.7735748, -26.5909290, 51.2852364, -73.9745407, 70.3645020
2: -12.1247120, 45.3733902, -14.2547655, 52.9671364, -65.0918503, 59.6281548
3: -25.8433571, 54.3873215, -30.3973598, 63.9421768, -89.7855377, 84.7846832
4: -15.5003433, 44.7416840, -18.3203106, 52.3190193, -67.8193588, 63.0619965

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 49

## BFS NS instance: NS_A1_B2_A2_B2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -68.4081573, 129.1255341, -82.3133698, 156.5305328, -224.9386902, 211.4389038
1: -23.2005672, 44.7357864, -28.0891361, 54.0074615, -77.2080231, 72.8249207
2: -12.3810244, 46.4367371, -14.9534559, 55.9817657, -68.3627930, 61.3901939
3: -26.4162025, 55.6006088, -31.8437176, 67.1011276, -93.5173264, 87.4443283
4: -15.8076572, 45.8427429, -19.1258888, 55.2833595, -71.0910187, 64.9686279

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A1_B2_A2_B2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -71.4678955, 134.8689880, -78.3574066, 147.7373962, -219.2052917, 213.2263947
1: -24.2386436, 46.6459732, -26.5852623, 51.0275879, -75.2662048, 73.2312317
2: -12.9322739, 48.4183350, -14.2363129, 53.0023880, -65.9346619, 62.6546478
3: -27.5348587, 57.9817047, -30.2151489, 63.5047493, -91.0395966, 88.1968536
4: -16.5068722, 47.7892151, -18.1454506, 52.4693222, -68.9761963, 65.9346619

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_B2_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_B2_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1478063, upper bound: 71.1454574
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1478063, upper bound: 71.1524771
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A1_A2_A2

### Backsubstitution after applying NS history:
0: -77.6091843, 147.8448486, -65.8488083, 123.1608505, -200.7700348, 213.6936646
1: -26.5033360, 51.0458412, -22.2078342, 42.6931381, -69.1964722, 73.2536545
2: -14.0914383, 52.9744072, -11.8839025, 44.2537460, -58.3451843, 64.8583069
3: -30.0887833, 63.4066505, -25.3091488, 53.0345116, -83.1232910, 88.7157974
4: -18.0381680, 52.2652740, -15.1839027, 43.6456490, -61.6838150, 67.4491730

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_B1_A2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_B1_A2_A1_A2_A2_B1

### Relational analysis result of NS_A2_B1_B1_A2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1456874, upper bound: 71.1507905
time: 0.88 seconds

## Relational analysis of NS_A2_B1_B1_A2_A1_A2_A2_B2

### Relational analysis result of NS_A2_B1_B1_A2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1456874, upper bound: 71.1507905
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -78.3574066, 147.7373962, -68.9923019, 130.8290253, -209.1864319, 216.7297058
1: -26.5852623, 51.0275879, -23.4937248, 45.3185081, -71.9037704, 74.5212860
2: -14.2363129, 53.0023880, -12.4724770, 46.9793510, -61.2156639, 65.4748611
3: -30.2151489, 63.5047493, -26.6935539, 56.2177467, -86.4328918, 90.1982803
4: -18.1454506, 52.4693222, -15.9719543, 46.2474632, -64.3929138, 68.4412766

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_B2_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B2_A2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B2_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_B2_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_B2_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_B2_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_B2_A2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_B2_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_B2_A2_A2_A2_B1_B1

### Relational analysis result of NS_A2_B1_B2_A2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1436418, upper bound: 71.1501280
time: 1.27 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B2_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_B2_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 49

## BFS NS instance: NS_A2_B1_B2_A2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -78.3574066, 147.7373962, -68.4081573, 129.1255341, -207.4829407, 216.1455536
1: -26.5852623, 51.0275879, -23.2005672, 44.7357864, -71.3210449, 74.2281342
2: -14.2363129, 53.0023880, -12.3810244, 46.4367371, -60.6730499, 65.3834152
3: -30.2151489, 63.5047493, -26.4162025, 55.6006088, -85.8157578, 89.9209290
4: -18.1454506, 52.4693222, -15.8076572, 45.8427429, -63.9881935, 68.2769775

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_B2_A2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B2_A2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B2_A2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_B2_A2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_B2_A2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B2_A2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_B2_A2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_B2_A2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_B2_A2_A2_A2_B2_B1

### Relational analysis result of NS_A2_B1_B2_A2_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1434169, upper bound: 71.1456390
time: 1.62 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_B2_A2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_B2_A2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 23

## BFS NS instance: NS_A2_B2_A2_B1_B1_B1_B1

### Backsubstitution after applying NS history:
0: -80.1428452, 150.4696808, -55.1105690, 105.4298553, -185.5726929, 205.5802460
1: -27.0836239, 52.0189972, -18.7954273, 36.1542282, -63.2378540, 70.8144226
2: -14.5107174, 53.9554482, -10.0173635, 37.5091858, -52.0199051, 63.9728088
3: -30.7238712, 64.7130814, -21.3924236, 44.9649696, -75.6888428, 86.1055069
4: -18.5133934, 53.3834190, -12.8451662, 36.9872589, -55.5006523, 66.2285843

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B1_B1_B1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1454696, upper bound: 71.1514518
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_B1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1454696, upper bound: 71.1517470
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B1_B1_B2

### Backsubstitution after applying NS history:
0: -83.2681198, 156.8382263, -68.7347946, 132.0270386, -215.2951660, 225.5730133
1: -28.2571011, 54.2695580, -23.4756031, 44.9308968, -73.1879730, 77.7451630
2: -15.1173201, 56.2637787, -12.6002512, 46.3624878, -61.4798088, 68.8640213
3: -32.0387726, 67.5293961, -26.8194199, 56.0100746, -88.0488358, 94.3487930
4: -19.2889595, 55.6473083, -16.1925220, 45.7353020, -65.0242462, 71.8398285

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.98 + 416.12 = 420.10 seconds
